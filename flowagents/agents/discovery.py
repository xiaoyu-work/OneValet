"""
Agent Discovery - Auto-discover @flowagent decorated agent classes

This module provides functionality to scan Python modules and packages
for agent classes decorated with @flowagent.

Usage:
    from flowagents.agents import AgentDiscovery, discover_agents

    # Scan a module
    discovery = AgentDiscovery()
    count = discovery.scan_module("myapp.agents.email")

    # Scan a package recursively
    count = discovery.scan_package("myapp.agents")

    # Scan multiple paths
    count = discovery.scan_paths(["myapp.agents", "myapp.custom_agents"])

    # Get discovered agents
    agents = discovery.get_discovered_agents()

    # Convenience function
    agents = discover_agents("myapp.agents")
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Optional, Type

from .decorator import AgentMetadata, AGENT_REGISTRY, get_agent_metadata

logger = logging.getLogger(__name__)


class AgentDiscovery:
    """
    Auto-discover and collect @flowagent decorated agent classes.

    This class scans Python modules and packages for classes decorated
    with @flowagent and collects them for registration with the orchestrator.

    Usage:
        discovery = AgentDiscovery()
        discovery.scan_package("myapp.agents")
        agents = discovery.get_discovered_agents()
    """

    def __init__(self):
        """Initialize agent discovery"""
        self._discovered_agents: Dict[str, AgentMetadata] = {}

    def scan_module(self, module_path: str) -> int:
        """
        Scan a module for @flowagent decorated classes.

        Args:
            module_path: Dot-separated module path (e.g., "myapp.agents.email")

        Returns:
            Number of agents discovered in this module
        """
        count = 0

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.warning(f"Failed to import module {module_path}: {e}")
            return 0

        # Scan all classes in the module
        for name in dir(module):
            obj = getattr(module, name)

            # Check if it's a class with flowagent metadata
            if inspect.isclass(obj):
                metadata = get_agent_metadata(obj)
                if metadata is not None:
                    # Only add if not already discovered
                    if metadata.name not in self._discovered_agents:
                        self._discovered_agents[metadata.name] = metadata
                        count += 1
                        logger.debug(
                            f"Discovered agent: {metadata.name} from {module_path}"
                        )

        return count

    def scan_package(self, package_path: str, recursive: bool = True) -> int:
        """
        Scan a package and all submodules for @flowagent decorated classes.

        Args:
            package_path: Dot-separated package path (e.g., "myapp.agents")
            recursive: Whether to scan subpackages (default: True)

        Returns:
            Number of agents discovered
        """
        count = 0

        try:
            package = importlib.import_module(package_path)
        except ImportError as e:
            logger.warning(f"Failed to import package {package_path}: {e}")
            return 0

        # Scan the package itself
        count += self.scan_module(package_path)

        # Scan submodules if package has __path__
        if hasattr(package, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(
                package.__path__,
                prefix=package_path + ".",
                onerror=lambda x: None,
            ):
                if recursive or not ispkg:
                    count += self.scan_module(modname)

        return count

    def scan_paths(self, paths: List[str]) -> int:
        """
        Scan multiple module/package paths for agents.

        Args:
            paths: List of module/package paths to scan

        Returns:
            Total number of agents discovered
        """
        count = 0
        for path in paths:
            count += self.scan_package(path)
        return count

    def get_discovered_agents(self) -> Dict[str, AgentMetadata]:
        """
        Get all discovered agents.

        Returns:
            Dictionary mapping agent name to AgentMetadata
        """
        return dict(self._discovered_agents)

    def get_agent(self, name: str) -> Optional[AgentMetadata]:
        """
        Get a specific discovered agent by name.

        Args:
            name: Agent name

        Returns:
            AgentMetadata or None if not found
        """
        return self._discovered_agents.get(name)

    def get_agent_names(self) -> List[str]:
        """
        Get list of discovered agent names.

        Returns:
            List of agent names
        """
        return list(self._discovered_agents.keys())

    def clear(self) -> None:
        """Clear all discovered agents"""
        self._discovered_agents.clear()

    def sync_from_global_registry(self) -> int:
        """
        Sync agents from the global AGENT_REGISTRY.

        This is useful when agents are auto-registered via the decorator
        but haven't been explicitly scanned.

        Returns:
            Number of agents added from global registry
        """
        count = 0
        for name, metadata in AGENT_REGISTRY.items():
            if name not in self._discovered_agents:
                self._discovered_agents[name] = metadata
                count += 1
        return count


def discover_agents(package_path: str, recursive: bool = True) -> Dict[str, AgentMetadata]:
    """
    Convenience function to discover agents from a package.

    Args:
        package_path: Package path to scan
        recursive: Whether to scan recursively

    Returns:
        Dictionary of discovered agents
    """
    discovery = AgentDiscovery()
    discovery.scan_package(package_path, recursive=recursive)
    discovery.sync_from_global_registry()
    return discovery.get_discovered_agents()


def discover_agents_from_paths(paths: List[str]) -> Dict[str, AgentMetadata]:
    """
    Convenience function to discover agents from multiple paths.

    Args:
        paths: List of package/module paths to scan

    Returns:
        Dictionary of discovered agents
    """
    discovery = AgentDiscovery()
    discovery.scan_paths(paths)
    discovery.sync_from_global_registry()
    return discovery.get_discovered_agents()


def get_global_registry() -> Dict[str, AgentMetadata]:
    """
    Get the global agent registry.

    This contains all agents that were auto-registered via the @flowagent
    decorator with auto_register=True.

    Returns:
        Dictionary mapping agent name to AgentMetadata
    """
    return dict(AGENT_REGISTRY)
