import os, json

if __name__ == "__main__":
    STIX_FOLDER = "sample_stix"
    ENTERPRISE_FILE = "mitre/enterprise-attack.json"
    
    tactics = set()
    for filename in os.listdir(STIX_FOLDER):
        if filename.endswith(".json"):
            with open(os.path.join(STIX_FOLDER, filename), "r", encoding='utf8') as f:
                data = json.load(f)
                for obj in data.get("objects", []):
                    if obj.get("type") == "attack-pattern":
                        for kill_chain_phase in obj.get("kill_chain_phases", []):
                            if kill_chain_phase.get("kill_chain_name") == "mitre-attack":
                                tactics.add(kill_chain_phase.get("phase_name"))

    # Load the enterprise-attack.json file to get the list of tactics
    enterprise_tactics = set()
    with open(ENTERPRISE_FILE, "r", encoding='utf8') as f:
        data = json.load(f)
        for obj in data.get("objects", []):
            if obj.get("type") == "x-mitre-tactic":
                enterprise_tactics.add(obj.get("x_mitre_shortname"))
    
    print(f"Number of unique tactics: {len(tactics)}")
    print(f"Tactics: {'\n'.join(sorted(tactics))}")

    difference_enterprise_stix = enterprise_tactics - tactics
    
    print(f"\nTactics in enterprise-attack.json but not in STIX files: {'\n'.join(sorted(difference_enterprise_stix))}")