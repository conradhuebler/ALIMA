import requests
import rdflib
from rdflib import Graph, Namespace

def get_dnb_classification(gnd_id):
    """
    Liest Klassifikationsinformationen aus den RDF-Daten der DNB.

    Args:
        gnd_id (str): GND-ID des Eintrags

    Returns:
        dict: Dictionary mit Klassifikationsinformationen
    """
    try:
        url = f"https://d-nb.info/gnd/{gnd_id}/about/lds"
        response = requests.get(url)
        response.raise_for_status()

        # Parse RDF
        g = Graph()
        g.parse(data=response.text, format='turtle')

        # Define namespaces
        GNDO = Namespace("https://d-nb.info/standards/elementset/gnd#")
        RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

        # Initialize result dictionary
        result = {
            'types': [],
            'category': None,
            'preferred_name': None,
            'ddc': [],
            'gnd_subject_categories': []
        }

        # Get entity URI
        entity_uri = rdflib.URIRef(f"https://d-nb.info/gnd/{gnd_id}")

        # Get all RDF types
        for type_triple in g.triples((entity_uri, RDF.type, None)):
            type_uri = str(type_triple[2])
            if 'standards/elementset/gnd#' in type_uri:
                type_name = type_uri.split('#')[1]
                result['types'].append(type_name)

                # Determine main category
                if 'DifferentiatedPerson' in type_name or 'Person' in type_name:
                    result['category'] = 'Person'
                elif 'Work' in type_name:
                    result['category'] = 'Work'
                elif any(keyword in type_name for keyword in ['SubjectHeading', 'Heading', 'Nomenclature']):
                    result['category'] = 'SubjectHeading'
                elif any(keyword in type_name for keyword in ['PlaceOrGeographic', 'Country', 'Territory']):
                    result['category'] = 'PlaceOrGeographic'

        # Get preferred name based on category
        name_predicates = []
        if result['category'] == 'Person':
            name_predicates.append(GNDO.preferredNameForThePerson)
        elif result['category'] == 'Work':
            name_predicates.append(GNDO.preferredNameForTheWork)
        elif result['category'] == 'SubjectHeading':
            name_predicates.append(GNDO.preferredNameForTheSubjectHeading)
        elif result['category'] == 'PlaceOrGeographic':
            name_predicates.append(GNDO.preferredNameForThePlaceOrGeographicName)

        for predicate in name_predicates:
            for name in g.triples((entity_uri, predicate, None)):
                result['preferred_name'] = str(name[2])
                if result['preferred_name']:
                    break
            if result['preferred_name']:
                break

        # Get DDC with determinacy
        for pred in [GNDO.relatedDdcWithDegreeOfDeterminacy1,
                    GNDO.relatedDdcWithDegreeOfDeterminacy2,
                    GNDO.relatedDdcWithDegreeOfDeterminacy3,
                    GNDO.relatedDdcWithDegreeOfDeterminacy4]:
            for ddc_triple in g.triples((entity_uri, pred, None)):
                ddc_uri = str(ddc_triple[2])
                ddc_code = ddc_uri.replace('http://dewey.info/class/', '').rstrip('/')
                determinacy = str(pred)[-1]
                result['ddc'].append({
                    'code': ddc_code,
                    'determinancy': determinacy
                })

        # Get GND subject categories
        for subj_cat in g.triples((entity_uri, GNDO.gndSubjectCategory, None)):
            category_uri = str(subj_cat[2])
            category_code = category_uri.split('#')[-1]
            result['gnd_subject_categories'].append(category_code)

        return result

    except Exception as e:
        print(f"Fehler beim Abrufen der DNB-Informationen für {gnd_id}: {str(e)}")
        return None

# Beispielverwendung:
#gnd_ids = ["118548018", "4064784-5", "4027833-5", "4099365-6"]  # Heine, Wasserstoff, Italien, Sommernachtstraum
#for gnd_id in gnd_ids:
#    result = get_dnb_classification(gnd_id)
#    print(f"\nGND-ID: {gnd_id}")
#    print(f"Name: {result['preferred_name']}")
#    print(f"Typen: {result['types']}")
#    print(f"Kategorie: {result['category']}")
#    print(f"DDC: {result['ddc']}")
#    print(f"GND-Sachgruppen: {result['gnd_subject_categories']}")
