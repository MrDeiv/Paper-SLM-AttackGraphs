import os, json

if __name__ == "__main__":
    STIX_FOLDER = "sample_stix"
    ENTERPRISE_FILE = "mitre/enterprise-attack.json"
    
    techniques = set()
    for filename in os.listdir(STIX_FOLDER):
        if filename.endswith(".json"):
            with open(os.path.join(STIX_FOLDER, filename), "r", encoding='utf8') as f:
                data = json.load(f)
                for obj in data.get("objects", []):
                    if obj.get("type") == "attack-pattern":
                        # get name
                        name = obj.get("name")
                        # remove "as used by ..."
                        if "as used by" in name:
                            name = name.split("as used by")[0].strip()
                        
                        if ":" not in name:
                            techniques.add(name)
    
    # Load the enterprise-attack.json file to get the list of techniques
    enterprise_techniques = set()
    with open(ENTERPRISE_FILE, "r", encoding='utf8') as f:
        data = json.load(f)
        for obj in data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                # get name
                name = obj.get("name")
                enterprise_techniques.add(name)
                
    print(f"Number of unique techniques: {len(techniques)}")
    print(f"\nNumber of unique techniques in enterprise-attack.json: {len(enterprise_techniques)}")
    
    difference_enterprise_stix = enterprise_techniques - techniques
    print(f"\nNumber of techniques in enterprise-attack.json but not in STIX files: {len(difference_enterprise_stix)}")