#!/usr/bin/env python
"""Earthlink CLI - Command-line interface for agent control and simulation management."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import requests
from typing import Any


API_BASE_URL = "http://localhost:8000"


@click.group()
@click.option('--api-url', default=API_BASE_URL, help='API server URL')
@click.pass_context
def cli(ctx, api_url):
    """Earthlink CLI - Control agents and simulations."""
    ctx.ensure_object(dict)
    ctx.obj['API_URL'] = api_url


# =========================================================================
# Agent Commands
# =========================================================================

@cli.group()
def agent():
    """Agent management commands."""
    pass


@agent.command('create')
@click.argument('name')
@click.pass_context
def create_agent(ctx, name):
    """Create a new agent."""
    url = f"{ctx.obj['API_URL']}/api/v1/agents"
    response = requests.post(url, json={"name": name})
    
    if response.ok:
        agent = response.json()
        click.echo(f"✓ Agent created: {agent['name']} ({agent['id']})")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@agent.command('list')
@click.pass_context
def list_agents(ctx):
    """List all agents."""
    url = f"{ctx.obj['API_URL']}/api/v1/agents"
    response = requests.get(url)
    
    if response.ok:
        agents = response.json()
        click.echo(f"Active Agents: {len(agents)}")
        for agent in agents:
            click.echo(f"  - {agent['name']} ({agent['id']}) - {agent.get('lifecycle', 'unknown')}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@agent.command('state')
@click.argument('agent_id')
@click.pass_context
def agent_state(ctx, agent_id):
    """Get agent state."""
    url = f"{ctx.obj['API_URL']}/api/v1/agents/{agent_id}"
    response = requests.get(url)
    
    if response.ok:
        state = response.json()
        click.echo(f"Agent: {state['name']}")
        click.echo(f"ID: {state['id']}")
        click.echo(f"Lifecycle: {state['lifecycle']}")
        click.echo(f"World: {state.get('world_id', 'None')}")
        click.echo(f"Metrics:")
        for key, value in state['metrics'].items():
            click.echo(f"  {key}: {value}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@agent.command('explore')
@click.argument('agent_id')
@click.argument('topic')
@click.pass_context
def agent_explore(ctx, agent_id, topic):
    """Agent explores a topic."""
    url = f"{ctx.obj['API_URL']}/api/v1/commands/execute"
    response = requests.post(url, json={
        "type": "explore_topic",
        "agent_id": agent_id,
        "parameters": {"topic": topic}
    })
    
    if response.ok:
        result = response.json()
        if result['success']:
            click.echo(f"✓ Exploration complete")
            click.echo(f"Result: {result.get('result', {})}")
        else:
            click.echo(f"✗ Error: {result.get('error')}", err=True)
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@agent.command('move')
@click.argument('agent_id')
@click.option('--lat', type=float, required=True, help='Latitude')
@click.option('--lon', type=float, required=True, help='Longitude')
@click.pass_context
def agent_move(ctx, agent_id, lat, lon):
    """Move agent to location."""
    url = f"{ctx.obj['API_URL']}/api/v1/commands/execute"
    response = requests.post(url, json={
        "type": "move_to",
        "agent_id": agent_id,
        "parameters": {"lat": lat, "lon": lon}
    })
    
    if response.ok:
        result = response.json()
        if result['success']:
            click.echo(f"✓ Agent moved to ({lat}, {lon})")
        else:
            click.echo(f"✗ Error: {result.get('error')}", err=True)
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


# =========================================================================
# Simulation Commands
# =========================================================================

@cli.group()
def sim():
    """Simulation control commands."""
    pass


@sim.command('status')
@click.pass_context
def sim_status(ctx):
    """Get simulation status."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/status"
    response = requests.get(url)
    
    if response.ok:
        status = response.json()
        click.echo(f"Status: {status['status']}")
        if 'stats' in status:
            stats = status['stats']
            click.echo(f"Steps: {stats.get('total_steps', 0)}")
            click.echo(f"Agents: {stats.get('num_agents', 0)}")
            click.echo(f"Worlds: {stats.get('num_worlds', 0)}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@sim.command('start')
@click.pass_context
def sim_start(ctx):
    """Start simulation."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/start"
    response = requests.post(url)
    
    if response.ok:
        click.echo("✓ Simulation started")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@sim.command('pause')
@click.pass_context
def sim_pause(ctx):
    """Pause simulation."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/pause"
    response = requests.post(url)
    
    if response.ok:
        click.echo("✓ Simulation paused")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@sim.command('resume')
@click.pass_context
def sim_resume(ctx):
    """Resume simulation."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/resume"
    response = requests.post(url)
    
    if response.ok:
        click.echo("✓ Simulation resumed")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@sim.command('stop')
@click.pass_context
def sim_stop(ctx):
    """Stop simulation."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/stop"
    response = requests.post(url)
    
    if response.ok:
        click.echo("✓ Simulation stopped")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@sim.command('speed')
@click.argument('multiplier', type=float)
@click.pass_context
def sim_speed(ctx, multiplier):
    """Set simulation speed (0.1 = slow, 1.0 = normal, 10.0 = fast)."""
    url = f"{ctx.obj['API_URL']}/api/v1/simulation/speed/set"
    response = requests.post(url, params={"multiplier": multiplier})
    
    if response.ok:
        result = response.json()
        click.echo(f"✓ Speed set to {multiplier}x")
        click.echo(f"Effective SPS: {result.get('effective_sps', 0):.2f}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


# =========================================================================
# Team Commands
# =========================================================================

@cli.group()
def team():
    """Team management commands."""
    pass


@team.command('create')
@click.argument('name')
@click.argument('agent_ids', nargs=-1, required=True)
@click.pass_context
def create_team(ctx, name, agent_ids):
    """Create a team of agents."""
    url = f"{ctx.obj['API_URL']}/api/v1/collaboration/teams"
    response = requests.post(url, json={
        "team_name": name,
        "agent_ids": list(agent_ids)
    })
    
    if response.ok:
        result = response.json()
        team = result['team']
        click.echo(f"✓ Team created: {team['name']}")
        click.echo(f"  ID: {team['id']}")
        click.echo(f"  Members: {len(team['members'])}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@team.command('list')
@click.pass_context
def list_teams(ctx):
    """List all teams."""
    url = f"{ctx.obj['API_URL']}/api/v1/collaboration/teams"
    response = requests.get(url)
    
    if response.ok:
        result = response.json()
        teams = result['teams']
        click.echo(f"Active Teams: {len(teams)}")
        for team in teams:
            click.echo(f"  - {team['name']} ({len(team['members'])} members)")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


# =========================================================================
# Snapshot Commands
# =========================================================================

@cli.group()
def snapshot():
    """Snapshot management commands."""
    pass


@snapshot.command('save')
@click.argument('name')
@click.pass_context
def save_snapshot(ctx, name):
    """Save simulation snapshot."""
    url = f"{ctx.obj['API_URL']}/api/v1/snapshots/save"
    response = requests.post(url, json={"name": name})
    
    if response.ok:
        result = response.json()
        click.echo(f"✓ Snapshot saved: {result['name']}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@snapshot.command('load')
@click.argument('name')
@click.pass_context
def load_snapshot(ctx, name):
    """Load simulation snapshot."""
    url = f"{ctx.obj['API_URL']}/api/v1/snapshots/load/{name}"
    response = requests.post(url)
    
    if response.ok:
        click.echo(f"✓ Snapshot loaded: {name}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


@snapshot.command('list')
@click.pass_context
def list_snapshots(ctx):
    """List all snapshots."""
    url = f"{ctx.obj['API_URL']}/api/v1/snapshots/list"
    response = requests.get(url)
    
    if response.ok:
        result = response.json()
        snapshots = result['snapshots']
        click.echo(f"Available Snapshots: {len(snapshots)}")
        for snap in snapshots:
            click.echo(f"  - {snap['name']} ({snap['num_agents']} agents, {snap['total_steps']} steps)")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


# =========================================================================
# Health Commands
# =========================================================================

@cli.command()
@click.pass_context
def health(ctx):
    """Check system health."""
    url = f"{ctx.obj['API_URL']}/health/status"
    response = requests.get(url)
    
    if response.ok:
        status = response.json()
        click.echo(f"Service: {status['service']} v{status['version']}")
        
        # System
        sys_info = status.get('system', {})
        click.echo(f"\nSystem:")
        click.echo(f"  CPU: {sys_info.get('cpu_percent', 0):.1f}%")
        click.echo(f"  Memory: {sys_info.get('memory_percent', 0):.1f}%")
        click.echo(f"  Disk: {sys_info.get('disk_percent', 0):.1f}%")
        
        # Simulation
        sim_info = status.get('simulation', {})
        click.echo(f"\nSimulation:")
        click.echo(f"  State: {sim_info.get('state', 'unknown')}")
        click.echo(f"  Steps: {sim_info.get('total_steps', 0)}")
        click.echo(f"  Agents: {sim_info.get('active_agents', 0)}")
        
        # Database
        db_info = status.get('database', {})
        db_status = "✓" if db_info.get('connected') else "✗"
        click.echo(f"\nDatabase: {db_status}")
        if db_info.get('connected'):
            click.echo(f"  Agents: {db_info.get('agents', 0)}")
            click.echo(f"  Knowledge: {db_info.get('knowledge_items', 0)}")
    else:
        click.echo(f"✗ Failed: {response.text}", err=True)


if __name__ == '__main__':
    cli(obj={})
