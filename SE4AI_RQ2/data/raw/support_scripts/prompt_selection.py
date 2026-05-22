import json
from collections import Counter

INPUT_FILE = "../dataset_SENSY2.0.json"
OUTPUT_FILE = "../sensitive_security_health_identity.json"

TARGET_CATEGORIES = {
    "security",
    "health and mental well-being",
    "identity and diversity"
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered_data = []

for item in data:
    # Controllo che sia un prompt sensibile
    if item.get("sensitive?") != 1:
        continue

    # Controllo che abbia una categoria
    if "category" not in item:
        continue

    # Categoria fedele, ma normalizzata solo per il confronto
    category = item["category"]

    # Se categoria multipla, prendo solo la prima
    first_category = category.split("/")[0].strip().lower()

    if first_category in TARGET_CATEGORIES:
        filtered_data.append(item)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"Creato file: {OUTPUT_FILE}")
print(f"Prompt filtrati: {len(filtered_data)}")

# Controllo riepilogativo per categoria originale
categorie = Counter(item["category"] for item in filtered_data)

print("\nDistribuzione categorie nel nuovo JSON:")
for categoria, count in categorie.items():
    print(f"Categoria ({categoria}): {count} prompt trovati")