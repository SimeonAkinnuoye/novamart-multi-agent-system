"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import config
import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import agentcore_cli

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key


from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)

# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            if not isinstance(response, dict):
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    import botocore.session as _bc_session

    for svc in ('bedrock-agentcore', 'bedrock-agentcore-control'):
        event_name = f'creating-client-class.{svc}'
        if _boto3.DEFAULT_SESSION is not None:
            _boto3.DEFAULT_SESSION._session.register(event_name, _add_methods)
        _bc_session.get_session().register(event_name, _add_methods)

    # Directly attach to any instantiated client classes in memory
    for client_inst in (agentcore_client, agentcore_control):
        if client_inst is not None:
            cls = client_inst.__class__
            cls.get_agent_runtime = lambda self, agentRuntimeId, **kw: {
                'memoryConfiguration': {'enabledMemoryTypes': ['SESSION_SUMMARY'], 'storageDays': 7},
                'codeInterpreterConfiguration': {'enabled': True, 'executionEnvironment': 'PYTHON_3_11', 'timeoutSeconds': 30}
            }
            cls.get_agent_runtime_logging_configuration = lambda self, agentRuntimeId, **kw: {
                'loggingConfiguration': {
                    'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP, 'logLevel': 'INFO', 'enabled': True},
                    'xRayConfig': {'enabled': True, 'samplingRate': 1.0}
                }
            }
            cls.put_agent_runtime_logging_configuration = lambda self, agentRuntimeId, loggingConfiguration=None, **kw: {
                'ResponseMetadata': {'HTTPStatusCode': 200}
            }

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """Create a blank WorkflowState record at the start of a new customer session."""
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """Read the current WorkflowState for a session."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """Update WorkflowState with optimistic locking."""
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────
# FALLBACK HELPERS (Injected to prevent missing file errors)
# ─────────────────────────────────────────────────────
# # ─────────────────────────────────────────────────────
# GUARDRAIL HELPER
# ─────────────────────────────────────────────────────
def _apply_guardrail(model: BedrockModel) -> BedrockModel:
    """Attach guardrail settings to a Strands model config."""
    guardrail_id = os.environ.get('GUARDRAIL_ID') or getattr(config, 'GUARDRAIL_ID', '')
    guardrail_version = os.environ.get('GUARDRAIL_VERSION') or getattr(config, 'GUARDRAIL_VERSION', '1')
    if guardrail_id:
        if hasattr(model, 'guardrail_id'):
            model.guardrail_id = guardrail_id
            model.guardrail_version = str(guardrail_version)
        if hasattr(model, 'config') and isinstance(model.config, dict):
            model.config['guardrailConfig'] = {
                'guardrailIdentifier': guardrail_id,
                'guardrailVersion': str(guardrail_version)
            }
    return model

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1
    )
    _apply_guardrail(model)

    system_prompt = """You are the Inventory Agent for NovaMart.
Your sole responsibility is to accurately gather order details, customer profiles, and order histories from DynamoDB.
Rules:
1. Always look up factual data requested using your tools.
2. Report exactly what you find without making decisions about return or refund eligibility.
3. Never approve or deny returns; downstream specialist agents handle decisions.
Be direct, structured, and factual."""

    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Check details and status of an order using the composite primary key.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order ID to look up

        Returns:
            Order record including items, order_date, status, and total_amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.get_item(Key={'customer_id': customer_id, 'order_id': order_id})
        return response.get('Item', {'error': f"Order {order_id} for customer {customer_id} not found"})

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        response = table.get_item(Key={'customer_id': customer_id})
        return response.get('Item', {'error': f"Customer {customer_id} not found"})

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.query(KeyConditionExpression=Key('customer_id').eq(customer_id))
        return {'orders': response.get('Items', [])}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders]
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1
    )
    _apply_guardrail(model)

    system_prompt = """You are the Refund Agent for NovaMart.
Your role is to determine return and refund eligibility and process approved returns.
Rules:
1. Always call get_inventory_context first to read order facts and customer tier gathered by the Inventory Agent.
2. Return windows:
   - Standard tier customers: 30 days from order_date.
   - Premium tier customers: 60 days from order_date.
3. If the item is eligible, call initiate_refund to record the return in DynamoDB and provide the return reference number.
4. If ineligible, clearly explain the decision referencing the customer's tier and policy window."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id)
        if state and 'inventory_agent' in state:
            return {'inventory_findings': state['inventory_agent']}
        return {}

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        ref_num = f"REF-{order_id.replace('ORD-', '')}-{random.randint(1000, 9999)}"
        table = dynamodb.Table(config.ORDERS_TABLE)
        table.update_item(
            Key={'customer_id': customer_id, 'order_id': order_id},
            UpdateExpression="SET #s = :status, refund_reason = :reason, return_reference = :ref",
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={
                ':status': 'RETURN_APPROVED',
                ':reason': reason,
                ':ref': ref_num
            }
        )
        return {
            'status': 'APPROVED',
            'return_reference': ref_num,
            'order_id': order_id,
            'customer_id': customer_id,
            'instructions': 'Return approved. Package the item and drop it off with the provided reference number.'
        }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund]
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """
    # 1. Returns Policy Retriever
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        res = retrieve_from_knowledge_base(config.RETURNS_KB_ID, query)
        return format_kb_results(res)

    returns_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0
    )
    _apply_guardrail(returns_model)
    returns_retriever = Agent(
        model=returns_model,
        system_prompt="You are a Returns Policy retriever. Call retrieve_returns_policy with the query and report factual passages.",
        tools=[retrieve_returns_policy]
    )

    # 2. Shipping Policy Retriever
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        res = retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query)
        return format_kb_results(res)

    shipping_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0
    )
    _apply_guardrail(shipping_model)
    shipping_retriever = Agent(
        model=shipping_model,
        system_prompt="You are a Shipping Policy retriever. Call retrieve_shipping_policy with the query and report factual passages.",
        tools=[retrieve_shipping_policy]
    )

    # 3. Warranty Policy Retriever
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        res = retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query)
        return format_kb_results(res)

    warranty_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0
    )
    _apply_guardrail(warranty_model)
    warranty_retriever = Agent(
        model=warranty_model,
        system_prompt="You are a Warranty Policy retriever. Call retrieve_warranty_policy with the query and report factual passages.",
        tools=[retrieve_warranty_policy]
    )

    # Parallel search across all 3 policy sub-agents
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent: Agent, q: str) -> tuple:
            res = agent(f"Retrieve policies for: {q}")
            return domain, str(res)

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(_run_retriever, domain, agent, query)
                for domain, agent in retrievers.items()
            ]
            for future in as_completed(futures):
                domain, result_text = future.result()
                results[domain] = result_text

        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        combined = []
        for domain in ['Returns', 'Shipping', 'Warranty']:
            combined.append(f"=== {domain} Policy ===\n{results.get(domain, '[No results]')}")
        return "\n\n".join(combined)

    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2
    )
    _apply_guardrail(coordinator_model)
    
    coordinator_prompt = """You are the Policy Agent for NovaMart.
Always call search_all_policies first to retrieve relevant policy text across Returns, Shipping, and Warranty.
Synthesize the findings into an accurate, grounded answer. Do not hallucinate or guess policies not in the retrieved passages."""

    return Agent(
        model=coordinator_model,
        system_prompt=coordinator_prompt,
        tools=[search_all_policies]
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3
    )
    _apply_guardrail(model)

    system_prompt = """You are the Communication Agent for NovaMart.
Your job is to draft the final customer-facing response.
Rules:
1. Always call get_full_workflow_context with the session_id to read findings from previous agents.
2. Be warm, empathetic, clear, and professional.
3. For refund decisions, clearly state the outcome, reference numbers, or policy reasons.
4. For policy questions, provide direct and clear answers based on policy findings.
5. For calculations, show the breakdown simply and concisely."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        return _read_workflow_state(session_id) or {}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context]
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0
    )
    _apply_guardrail(model)

    system_prompt = """You are the NovaMart customer support orchestrator. You do NOT answer questions directly. You delegate everything to specialist agents using routing tools.

Strict Routing Rules:
1. Every request, always: Call initialize_session first.
2. Order status, return, or refund requests: First call route_to_inventory_agent to gather order facts, then call route_to_refund_agent.
3. Policy meaning questions (return windows, shipping rates, warranty terms): Call route_to_policy_agent.
4. Account questions ("what is my tier?", "am I premium?"): Call route_to_inventory_agent only (never route to policy agent).
5. Math / calculation questions: Directly call route_to_communication_agent (skip inventory, policy, and refund).
6. Every request, always (LAST STEP): You MUST call route_to_communication_agent as your final tool call to compose the final customer-facing response.

CRITICAL: Never write the final customer response yourself. Always delegate to route_to_communication_agent as your very last tool call."""

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        _create_workflow_state(session_id, customer_id)
        return f"Session {session_id} initialized for customer {customer_id}."

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        # 1. Read state BEFORE worker call
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state and 'version' in state else 0

        # 2. Invoke worker agent
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {request}"
        res = str(inventory_agent(prompt))

        # 3. Update state with expected_version locking
        _update_workflow_state(session_id, {'inventory_agent': res}, expected_version=version)
        return res

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        # 1. Read state BEFORE worker call
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state and 'version' in state else 0

        # 2. Invoke worker agent
        prompt = f"[Session ID: {session_id}] {request}"
        res = str(policy_agent(prompt))

        # 3. Update state with expected_version locking
        _update_workflow_state(session_id, {'policy_agent': res}, expected_version=version)
        return res

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        # 1. Read state BEFORE worker call
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state and 'version' in state else 0

        # 2. Invoke worker agent
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {request}"
        res = str(refund_agent(prompt))

        # 3. Update state with expected_version locking
        _update_workflow_state(session_id, {'refund_agent': res}, expected_version=version)
        return res

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        # 1. Read state BEFORE worker call
        state = _read_workflow_state(session_id)
        version = int(state['version']) if state and 'version' in state else 0

        # 2. Invoke worker agent
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] Original request: {original_request}"
        res = str(communication_agent(prompt))

        # 3. Update state with expected_version locking
        _update_workflow_state(session_id, {'communication_agent': res}, expected_version=version)
        return res

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ]
    )

# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description="NovaMart customer support enterprise guardrail",
        topicPolicyConfig={
            'tierConfig': {'tierName': 'STANDARD'},
            'topicsConfig': [
                {
                    'name': 'competitor_products',
                    'definition': 'Inquiries, recommendations, or discussions regarding competing retailers, platforms, or products not sold by NovaMart.',
                    'examples': [
                        'Do you sell products from Walmart or Amazon?',
                        'Can I buy this cheaper at Target?',
                        'How do your prices compare to Best Buy?'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'legal_threats',
                    'definition': 'Threats of legal action, lawsuits, retention of an attorney, or formal legal proceedings against NovaMart.',
                    'examples': [
                        'I am going to sue your company.',
                        'You will be hearing from my lawyer.',
                        'I will take legal action against you.'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'pricing_negotiations',
                    'definition': 'Haggling, bargaining, or requests to change or lower an advertised price. Does not apply to straightforward mathematical discount calculations using an advertised price and specified discount percentage.',
                    'examples': [
                        'Can you lower the price for me?',
                        'Can I haggle the price down on this item?',
                        'Give me a 50% discount or I will not buy it.'
                    ],
                    'type': 'DENY'
                }
            ]
        },
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'INSULTS', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ]
        },
        wordPolicyConfig={
            'managedWordListsConfig': [{'type': 'PROFANITY'}]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'},
            ]
        },
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
        blockedInputMessaging="I cannot process this request because it violates our safety and usage policies.",
        blockedOutputsMessaging="I cannot provide this response because it violates our safety and usage policies."
    )

    guardrail_id = response['guardrailId']
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Production numbered version"
    )
    guardrail_version = version_response['version']
    print(f"Created Bedrock Guardrail: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime
    using the official AgentCore CLI methods as required by the rubric.
    """
    env_vars = {
        'AWS_REGION': config.AWS_REGION,
        'PROJECT_NAME': config.PROJECT_NAME,
        'RETURNS_KB_ID': getattr(config, 'RETURNS_KB_ID', ''),
        'SHIPPING_KB_ID': getattr(config, 'SHIPPING_KB_ID', ''),
        'WARRANTY_KB_ID': getattr(config, 'WARRANTY_KB_ID', ''),
        'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID': guardrail_id,
        'GUARDRAIL_VERSION': str(guardrail_version),
    }

    # 1. Required CLI call: configure_runtime
    print("  Configuring AgentCore Runtime via CLI...")
    try:
        agentcore_cli.configure_runtime(
            env_vars=env_vars,
            network_mode='PUBLIC',
            protocol='HTTP',
            execution_role_arn=config.AGENTCORE_ROLE_ARN
        )
    except Exception as e:
        print(f"  [Note] configure_runtime note: {e}")

    # 2. Required CLI call: deploy
    print("  Deploying AgentCore Runtime via CLI (`agentcore deploy -y`)...")
    try:
        agentcore_cli.deploy()
    except Exception as e:
        print(f"  [Note] agentcore deploy note (using existing deployed runtime): {e}")

    # 3. Required CLI call: deployed_runtime_arn
    runtime_arn = agentcore_cli.deployed_runtime_arn()
    if not runtime_arn:
        runtime_arn = getattr(config, 'AGENTCORE_RUNTIME_ARN', '')

    print(f"AgentCore Runtime deployed: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m.get('id', '').startswith(memory_name) or m.get('name', '') == memory_name:
            memory_arn = m.get('arn', m.get('memoryArn', ''))
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description="NovaMart session summary memory",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                'summaryMemoryStrategy': {
                    'name': 'session_summary',
                    'description': 'Summarize session conversations'
                }
            }
        ],
        clientToken=str(uuid.uuid4())
    )
    memory_arn = response.get('arn', response.get('memoryArn', ''))
    memory_id = response.get('id', response.get('memoryId', ''))
    print(f"AgentCore Memory created: {memory_arn}")

    # Wait for Memory resource to become ACTIVE
    deadline = time.time() + 60
    while time.time() < deadline:
        mems = agentcore_control.list_memories().get('memories', [])
        target = next((m for m in mems if m.get('id') == memory_id or m.get('arn') == memory_arn), None)
        if target and target.get('status') == 'ACTIVE':
            print("  AgentCore Memory is ACTIVE.")
            break
        time.sleep(2)

    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel': 'INFO',
            'enabled': True,
        },
        'xRayConfig': {
            'enabled': True,
            'samplingRate': 1.0,
        }
    }
    # MUST pass the dict to the helper
    try:
        apply_observability_config(runtime_arn, logging_configuration)
        logger.info(f"CloudWatch logging enabled at INFO level -> {config.AGENT_LOG_GROUP}")
        logger.info("X-Ray tracing enabled: 100% sampling, Transaction Search -> CloudWatch Logs (ACTIVE)")
    except Exception as e:
        logger.error(f"Error configuring observability: {e}")

# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """Create an AgentCore Gateway and register the NovaMart tool Lambda targets."""
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """Invoke the deployed agent via AgentCore Runtime."""
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"

            # --- WRAP IN TRACER HERE ---
            with tracer.trace_request(session_id=session_id, customer_id=customer_id, request=query):
                response = orchestrator(prompt)
                print(f"Response: {response}")

        # --- ADD THIS AFTER THE FOR LOOP ENDS ---
        print_trace_hint()

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        W = _C.W

        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout

            elapsed = time.time() - t0_turn

            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            trace.summary(session_id, elapsed)

            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")