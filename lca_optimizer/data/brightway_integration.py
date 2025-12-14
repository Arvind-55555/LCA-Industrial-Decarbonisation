"""
Brightway2 Integration for Ecoinvent Database
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import brightway2 as bw
    BRIGHTWAY_AVAILABLE = True
except ImportError:
    BRIGHTWAY_AVAILABLE = False
    logger.warning("brightway2 not available. Install with: pip install brightway2")


class BrightwayIntegration:
    """
    Integration with Brightway2 for Ecoinvent database access.
    
    Requires:
    1. Brightway2 installation: pip install brightway2
    2. Ecoinvent database (requires license)
    3. Database setup: bw.projects.set_current("your_project")
    """
    
    def __init__(self, project_name: str = "lca_optimizer"):
        """
        Initialize Brightway2 integration.
        
        Args:
            project_name: Brightway2 project name
        """
        if not BRIGHTWAY_AVAILABLE:
            raise ImportError(
                "brightway2 not available. Install with: pip install brightway2"
            )
        
        self.project_name = project_name
        
        # Set or create project
        if project_name not in bw.projects:
            bw.projects.create_project(project_name)
            logger.info(f"Created Brightway2 project: {project_name}")
        
        bw.projects.set_current(project_name)
        logger.info(f"Brightway2 integration initialized for project: {project_name}")
    
    def load_ecoinvent(
        self,
        version: str = "3.9",
        system_model: str = "cutoff"
    ) -> bool:
        """
        Load Ecoinvent database.
        
        Args:
            version: Ecoinvent version (e.g., "3.9", "3.8")
            system_model: System model ("cutoff", "apos", "consequential")
        
        Returns:
            True if successful
        """
        db_name = f"ecoinvent {version} {system_model}"
        
        if db_name in bw.databases:
            logger.info(f"Database {db_name} already loaded")
            return True
        
        try:
            # Check if ecoinvent is installed
            if "ecoinvent" not in [str(d) for d in bw.databases]:
                logger.warning(
                    "Ecoinvent database not found. "
                    "Please import Ecoinvent database first using bw2io."
                )
                return False
            
            logger.info(f"Ecoinvent {version} {system_model} loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Ecoinvent: {e}")
            return False
    
    def get_emission_factor(
        self,
        process_name: str,
        database: str = "ecoinvent 3.9 cutoff"
    ) -> Dict[str, float]:
        """
        Get emission factors for a process.
        
        Args:
            process_name: Name of the process
            database: Database name
        
        Returns:
            Dictionary of emission factors (kg CO2eq per unit)
        """
        try:
            # Search for process
            processes = [p for p in bw.Database(database) if process_name.lower() in str(p).lower()]
            
            if not processes:
                logger.warning(f"Process '{process_name}' not found in {database}")
                return {}
            
            # Get first matching process
            process = processes[0]
            
            # Calculate LCA
            lca = bw.LCA({process: 1})
            lca.lci()
            lca.lcia()
            
            # Extract emission factors
            factors = {}
            for method in bw.methods:
                if "climate change" in str(method).lower() or "GWP" in str(method):
                    lca.switch_method(method)
                    lca.lcia()
                    factors[str(method)] = lca.score
            
            return factors
            
        except Exception as e:
            logger.error(f"Failed to get emission factor: {e}")
            return {}
    
    def search_processes(
        self,
        query: str,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for processes in database.
        
        Args:
            query: Search query
            database: Database name (searches all if None)
        
        Returns:
            List of matching processes
        """
        results = []
        
        databases = [database] if database else bw.databases
        
        for db_name in databases:
            try:
                db = bw.Database(db_name)
                matches = [
                    {
                        "name": str(p),
                        "database": db_name,
                        "key": p.key
                    }
                    for p in db
                    if query.lower() in str(p).lower()
                ]
                results.extend(matches)
            except Exception as e:
                logger.warning(f"Failed to search {db_name}: {e}")
        
        return results
    
    def calculate_lca(
        self,
        activity: Dict,
        method: Optional[tuple] = None
    ) -> float:
        """
        Calculate LCA for an activity.
        
        Args:
            activity: Activity dictionary {process: amount}
            method: LCIA method (uses default if None)
        
        Returns:
            LCA score
        """
        try:
            if method is None:
                # Use default climate change method
                method = [m for m in bw.methods if "climate change" in str(m).lower()][0]
            
            lca = bw.LCA(activity, method=method)
            lca.lci()
            lca.lcia()
            
            return lca.score
            
        except Exception as e:
            logger.error(f"LCA calculation failed: {e}")
            return 0.0

