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
