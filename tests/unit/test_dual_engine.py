import pytest
import re
import json
import os
import sys

def test_dual_engine_model_defaults():
    from app.agent import HOST_MODEL_DEEP, HOST_MODEL_REFLEX, root_agent
    assert HOST_MODEL_DEEP == "gemini-3.5-flash"
    assert HOST_MODEL_REFLEX == "gemini-3.5-flash-lite"
    assert root_agent.model.model == "gemini-3.5-flash"

def test_co_browse_action_parsing():
    sample_response = """<<<CO_BROWSE_ACTION: {"type": "scroll_and_highlight", "targetText": "Clutch Global Leader"}>>>
We are recognized by Clutch as a global leader in software development."""
    
    matches = re.findall(r"<<<CO_BROWSE_ACTION:\s*(\{.*?\})\s*>>>", sample_response, re.DOTALL)
    assert len(matches) == 1
    action = json.loads(matches[0])
    assert action["type"] == "scroll_and_highlight"
    assert action["targetText"] == "Clutch Global Leader"

def test_skill_single_source_of_truth():
    from app.agent import agent_name, agent_description
    assert agent_name == "host_agent_test"
    assert "Gemini 3.5" in agent_description

def test_resolve_active_agent_embed_mode():
    from app.agent import resolve_active_agent
    ctx = {
        "mode": "chat_embed",
        "system_instruction": "Workspace: Zcoai\nPersona: Friendly hostess",
        "domDigest": {
            "title": "Zco Homepage",
            "markdownDigest": "## Pricing\nCustom mobile apps start at $10k."
        }
    }
    agent, app_name = resolve_active_agent(ctx, "Where is the pricing table?")
    assert "reflex" in agent.name
    assert agent.model.model == "gemini-3.5-flash-lite"
    assert "CO-BROWSE VISUAL ACTUATION INSTRUCTIONS" in agent.instruction
    assert "Zco Homepage" in agent.instruction
    assert "Pricing" in agent.instruction
    # Ensure no subagent delegation tools are attached
    tool_names = [getattr(t, "__name__", "") for t in agent.tools]
    assert "discover_agents" not in tool_names
    assert "consultAgent" not in tool_names
    assert "consult_agent" not in tool_names

def test_resolve_active_agent_subagent_fallback():
    from app.agent import resolve_active_agent
    ctx = {
        "mode": "chat_embed",
        "system_instruction": "Workspace: Zcoai"
    }
    agent, app_name = resolve_active_agent(ctx, "Please consult_agent knowledge_agent about patents")
    # Subagent keywords trigger Deliberation mode (root_agent with gemini-3.5-flash)
    assert "reflex" not in agent.name
    assert agent.model.model == "gemini-3.5-flash"
