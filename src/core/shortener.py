#!/usr/bin/env python3

def analyze_keywords(keywords):
    """
    Analysiert eine Liste von Schlagworten und identifiziert eigenständige
    Schlagworte sowie Schlagworte, die nur Teil längerer Ketten sind.
    
    Ein Schlagwort gilt als eigenständig, wenn es entweder:
    - In keiner längeren Kette enthalten ist ODER
    - Als individueller Eintrag in der ursprünglichen Liste vorkommt
    
    Args:
        keywords (list): Liste von Schlagworten und Schlagwortketten
        
    Returns:
        tuple: (eigenständige Schlagworte, Teil-Schlagworte)
    """
    # Entferne Duplikate, aber behalte die ursprüngliche Liste für späteren Vergleich
    original_unique = list(dict.fromkeys(keywords))
    
    # Sortiere nach Länge (Anzahl der Wörter), vom längsten zum kürzesten
    sorted_keywords = sorted(original_unique, key=lambda x: len(x.split()), reverse=True)
    
    # Identifiziere Schlagworte, die in anderen enthalten sind
    contained_in_others = set()
    for i, keyword in enumerate(sorted_keywords):
        for j, other in enumerate(sorted_keywords):
            if i != j and is_complete_substring(keyword, other):
                contained_in_others.add(keyword)
    
    # Ein Schlagwort ist eigenständig, wenn es als individueller Eintrag vorkommt
    standalone_keywords = original_unique.copy()
    
    # Ein Schlagwort ist nur Teil anderer, wenn es nie alleine vorkommt
    part_keywords = [kw for kw in contained_in_others if kw not in original_unique]
    
    # Wenn ein Wort sowohl eigenständig vorkommt als auch Teil einer Kette ist,
    # dann sollte es in der Liste der eigenständigen Schlagworte bleiben
    
    return standalone_keywords, part_keywords

def is_complete_substring(shorter, longer):
    """
    Prüft, ob shorter als vollständiges Schlagwort in longer vorkommt.
    Das heißt, alle Wörter von shorter müssen in der gleichen Reihenfolge in longer vorkommen.
    
    Args:
        shorter (str): Das möglicherweise enthaltene Schlagwort
        longer (str): Die möglicherweise umfassendere Schlagwortkette
        
    Returns:
        bool: True, wenn shorter als vollständiges Schlagwort in longer vorkommt
    """
    if shorter == longer:
        return False  # Ein Schlagwort ist nicht Teil seiner selbst
        
    # Zerlege die Strings in Wörter
    shorter_words = shorter.split()
    longer_words = longer.split()
    
    # Wenn shorter mehr Wörter hat als longer, kann es nicht enthalten sein
    if len(shorter_words) > len(longer_words):
        return False
        
    # Prüfe auf aufeinanderfolgende Übereinstimmung der Wörter
    for i in range(len(longer_words) - len(shorter_words) + 1):
        match = True
        for j in range(len(shorter_words)):
            if shorter_words[j] != longer_words[i + j]:
                match = False
                break
        if match:
            return True
            
    return False

def print_analysis(keywords):
    """
    Druckt eine ausführliche Analyse der Schlagworte.
    
    Args:
        keywords (list): Liste von Schlagworten und Schlagwortketten
    """
    print(f"Eingabe: {len(keywords)} Schlagworte/Schlagwortketten")
    for kw in keywords:
        print(f"  - {kw}")
    print()
    
    standalone, parts = analyze_keywords(keywords)
    
    print(f"Eigenständige Schlagworte ({len(standalone)}):")
    for kw in standalone:
        # Zeige an, ob dieses Schlagwort auch in längeren Ketten vorkommt
        in_longer = False
        for other in standalone:
            if kw != other and is_complete_substring(kw, other):
                in_longer = True
                break
        
        if in_longer:
            print(f"  - {kw} (kommt auch in längeren Ketten vor)")
        else:
            print(f"  - {kw}")
    print()
    
    if parts:
        print(f"Schlagworte, die nur Teil längerer Ketten sind ({len(parts)}):")
        for kw in parts:
            # Finde übergeordnete Ketten
            parent_chains = []
            for other in standalone:
                if kw != other and is_complete_substring(kw, other):
                    parent_chains.append(other)
            
            parent_str = ", ".join(parent_chains)
            print(f"  - {kw} (Teil von: {parent_str})")
    else:
        print("Keine Schlagworte gefunden, die nur Teil längerer Ketten sind.")

# Beispiel-Verwendung
if __name__ == "__main__":
    # Beispiel aus der Aufgabenstellung
    example_keywords = [
        "Molekulardynamik", 
        "Statistische Mechanik", 
        "Monte-Carlo-Simulation", 
        "Molekulardynamik Statistische Mechanik Monte-Carlo-Simulation", 
        "Molekulardynamik Statistische Mechanik", 
        "Lehrbuch", 
        "Statistische Mechanik"  # Duplikat zur Demonstration
    ]
    
    print("=== Analyse der Beispiel-Schlagworte ===")
    print_analysis(example_keywords)
    
    # Interaktiver Modus
    print("\n" + "="*50)
    print("Eigene Schlagworte analysieren? (j/n): ", end="")
    if input().lower().startswith("j"):
        print("\nGib Schlagworte ein (leere Zeile zum Beenden):")
        user_keywords = []
        while True:
            user_input = input("> ").strip()
            if not user_input:
                break
            user_keywords.append(user_input)
        
        if user_keywords:
            print("\n=== Analyse deiner Schlagworte ===")
            print_analysis(user_keywords)
        else:
            print("Keine Schlagworte eingegeben.")

