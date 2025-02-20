import requests
import rdflib
from rdflib import Graph, Namespace
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dnb_classification(gnd_id):
    """
    Liest Klassifikationsinformationen aus den RDF-Daten der DNB.
    
    Args:
        gnd_id (str): GND-ID des Eintrags
        
    Returns:
        dict: Dictionary mit Klassifikationsinformationen oder None bei Fehler
    """
    # Standardwerte für das Ergebnis
    result = {
        'types': [],
        'category': None,
        'preferred_name': None,
        'ddc': [],
        'gnd_subject_categories': [],
        'status': 'success',
        'error_message': None
    }
    
    try:
        # URL-Validierung
        if not gnd_id or not isinstance(gnd_id, str):
            raise ValueError(f"Ungültige GND-ID: {gnd_id}")
            
        url = f"https://d-nb.info/gnd/{gnd_id}/about/lds"
        
        # HTTP-Anfrage mit Timeout
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Fehler bei HTTP-Anfrage für GND {gnd_id}: {str(e)}")
            result.update({
                'status': 'error',
                'error_message': f"Netzwerkfehler: {str(e)}"
            })
            return result
        
        # RDF Parser
        try:
            g = Graph()
            g.parse(data=response.text, format='turtle')
        except Exception as e:
            logger.error(f"Fehler beim Parsen der RDF-Daten für GND {gnd_id}: {str(e)}")
            result.update({
                'status': 'error',
                'error_message': f"RDF-Parse-Fehler: {str(e)}"
            })
            return result
        
        # Define namespaces
        GNDO = Namespace("https://d-nb.info/standards/elementset/gnd#")
        RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        
        # Get entity URI
        entity_uri = rdflib.URIRef(f"https://d-nb.info/gnd/{gnd_id}")
        
        # Get all RDF types
        type_found = False
        for type_triple in g.triples((entity_uri, RDF.type, None)):
            type_uri = str(type_triple[2])
            if 'standards/elementset/gnd#' in type_uri:
                type_name = type_uri.split('#')[1]
                result['types'].append(type_name)
                type_found = True
                
                # Determine main category
                if 'DifferentiatedPerson' in type_name or 'Person' in type_name:
                    result['category'] = 'Person'
                elif 'Work' in type_name:
                    result['category'] = 'Work'
                elif any(keyword in type_name for keyword in ['SubjectHeading', 'Heading', 'Nomenclature']):
                    result['category'] = 'SubjectHeading'
                elif any(keyword in type_name for keyword in ['PlaceOrGeographic', 'Country', 'Territory']):
                    result['category'] = 'PlaceOrGeographic'
        
        if not type_found:
            logger.warning(f"Keine RDF-Typen gefunden für GND {gnd_id}")
        
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
        
        name_found = False
        for predicate in name_predicates:
            for name in g.triples((entity_uri, predicate, None)):
                result['preferred_name'] = str(name[2])
                name_found = True
                break
            if name_found:
                break
                
        if not name_found:
            logger.warning(f"Kein bevorzugter Name gefunden für GND {gnd_id}")
            result['preferred_name'] = f"Unbekannter Name (GND: {gnd_id})"
        
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
        logger.error(f"Unerwarteter Fehler bei GND {gnd_id}: {str(e)}")
        result.update({
            'status': 'error',
            'error_message': f"Unerwarteter Fehler: {str(e)}"
        })
        return result

# Beispielverwendung:
#ef safe_print_result(gnd_id):
#    result = get_dnb_classification(gnd_id)
#    if result:
#        print(f"\nGND-ID: {gnd_id}")
#        print(f"Status: {result['status']}")
#        if result['error_message']:
#            print(f"Fehler: {result['error_message']}")
#        else:
#            print(f"Name: {result['preferred_name']}")
#            print(f"Typen: {result['types']}")
#            print(f"Kategorie: {result['category']}")
#            print(f"DDC: {result['ddc']}")
#            print(f"GND-Sachgruppen: {result['gnd_subject_categories']}")
#    else:
#        print(f"\nKeine Daten für GND-ID: {gnd_id}")

# Test
#gnd_ids = ["118548018", "4064784-5", "4027833-5", "4099365-6", "invalid-id"]
#for gnd_id in gnd_ids:
#    safe_print_result(gnd_id)
