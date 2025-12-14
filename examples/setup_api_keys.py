#!/usr/bin/env python3
"""
Setup API Keys for Real Data Integration
Interactive script to help set up API keys
"""

import os
from pathlib import Path
from getpass import getpass


def setup_env_file():
    """Interactive setup for .env file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # Create .env from example if it doesn't exist
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("Created .env file from .env.example")
    
    print("\n" + "=" * 60)
    print("API Key Setup for LCA Optimizer")
    print("=" * 60)
    print("\nThis script will help you set up API keys for real data integration.")
    print("You can skip any API by pressing Enter.\n")
    
    # Read existing .env
    env_vars = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    # Electricity Maps
    print("1. Electricity Maps API")
    print("   Get your API key from: https://www.electricitymaps.com/")
    current = env_vars.get("ELECTRICITY_MAPS_API_KEY", "")
    new_key = input(f"   API Key [{current if current else 'not set'}]: ").strip()
    if new_key:
        env_vars["ELECTRICITY_MAPS_API_KEY"] = new_key
    
    # WattTime
    print("\n2. WattTime API")
    print("   Sign up at: https://www.watttime.org/")
    current_user = env_vars.get("WATTTIME_USERNAME", "")
    current_pass = env_vars.get("WATTTIME_PASSWORD", "")
    new_user = input(f"   Username [{current_user if current_user else 'not set'}]: ").strip()
    if new_user:
        env_vars["WATTTIME_USERNAME"] = new_user
        new_pass = getpass(f"   Password [{'(set)' if current_pass else 'not set'}]: ")
        if new_pass:
            env_vars["WATTTIME_PASSWORD"] = new_pass
    
    # ENTSO-E
    print("\n3. ENTSO-E Transparency Platform")
    print("   Get security token from: https://transparency.entsoe.eu/")
    current = env_vars.get("ENTSOE_SECURITY_TOKEN", "")
    new_token = input(f"   Security Token [{current if current else 'not set'}]: ").strip()
    if new_token:
        env_vars["ENTSOE_SECURITY_TOKEN"] = new_token
    
    # Write .env file
    print("\n" + "=" * 60)
    save = input("Save these settings to .env file? (y/n): ").strip().lower()
    
    if save == 'y':
        # Read template
        template = env_example.read_text() if env_example.exists() else ""
        
        # Update values
        lines = template.splitlines()
        updated_lines = []
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                key = line.split("=", 1)[0].strip()
                if key in env_vars:
                    updated_lines.append(f"{key}={env_vars[key]}")
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Add new variables
        existing_keys = {line.split("=")[0].strip() for line in lines if "=" in line}
        for key, value in env_vars.items():
            if key not in existing_keys:
                updated_lines.append(f"{key}={value}")
        
        env_file.write_text("\n".join(updated_lines))
        print("\n✅ Settings saved to .env file!")
        print("   Remember: .env is in .gitignore and won't be committed.")
    else:
        print("\nSettings not saved.")


if __name__ == "__main__":
    setup_env_file()

