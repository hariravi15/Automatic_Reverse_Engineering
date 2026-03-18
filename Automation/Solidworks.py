import requests
import json
import time
import os


API_ACCESS_KEY = 'on_kdmrk2iumIHsYp2FOvYyg'
API_SECRET_KEY = 'gKBYn2dfEptL8OFY83YYKD55jYd044crqRSguzfOWPnrpIuK'
BASE_URL = 'https://cad.onshape.com'
OUTPUT_DIR = r"C:\Users\hr73\PycharmProjects\Reverse_engineeering\Automation\CAD_Output"


def make_request(method, endpoint, headers=None, body=None):
    url = BASE_URL + endpoint
    auth = (API_ACCESS_KEY, API_SECRET_KEY)
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    try:
        response = requests.request(method, url, auth=auth, headers=headers, json=body)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err}")
        print(response.text)
        return None


def get_last_modified_document():
    response = make_request('GET', '/api/documents')
    if not response: return None

    docs = response.json().get('items', [])
    if not docs:
        print("No documents found.")
        return None

    docs.sort(key=lambda x: x['modifiedAt'], reverse=True)
    last_doc = docs[0]

    print(f"Found last modified document: {last_doc['name']} (ID: {last_doc['id']})")
    return last_doc


def get_first_exportable_element(doc_id, workspace_id):
    endpoint = f'/api/documents/d/{doc_id}/w/{workspace_id}/elements'
    response = make_request('GET', endpoint)
    if not response: return None
    elements = response.json()

    for el in elements:
        if el['elementType'] == 'PARTSTUDIO':
            return el
    for el in elements:
        if el['elementType'] == 'ASSEMBLY':
            return el

    return None


def export_element(doc_id, workspace_id, element):
    el_id = element['id']
    el_type = element['elementType']
    name = element['name']

    print(f"Preparing to export '{name}' ({el_type})...")
    type_path = "partstudios" if el_type == 'PARTSTUDIO' else "assemblies"
    endpoint = f'/api/{type_path}/d/{doc_id}/w/{workspace_id}/e/{el_id}/translations'

    payload = {
        "formatName": "SOLIDWORKS",
        "storeInDocument": False,
        "versionString": "33.0"
    }

    response = make_request('POST', endpoint, body=payload)
    if not response: return
    translation_id = response.json()['id']
    print(f"Translation started (ID: {translation_id}). Polling for completion...")

    while True:
        status_res = make_request('GET', f'/api/translations/{translation_id}')
        if not status_res: break
        state = status_res.json()['requestState']
        if state == 'DONE':
            print("Translation complete.")
            result_id = status_res.json()['resultExternalDataIds'][0]
            if el_type == 'ASSEMBLY':
                ext = "zip"
            elif payload['formatName'] == "STEP":
                ext = "step"
            else:
                ext = "sldprt"
            download_file(doc_id, result_id, name, ext)
            break
        time.sleep(2)  # Wait 2 seconds


def download_file(doc_id, external_data_id, base_name, extension):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    filename = f"{base_name.replace(' ', '_')}.{extension}"
    filepath = os.path.join(OUTPUT_DIR, filename)
    url = f"{BASE_URL}/api/documents/d/{doc_id}/externaldata/{external_data_id}"
    response = requests.get(url, auth=(API_ACCESS_KEY, API_SECRET_KEY), stream=True)

    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"SUCCESS: File saved to -> {filepath}")
    else:
        print(f"Failed to download file. Status: {response.status_code}")
        print(f"URL attempted: {url}")

if __name__ == "__main__":
    print("--- Onshape Last Design Exporter ---")
    doc = get_last_modified_document()
    if doc:
        workspace_id = doc['defaultWorkspace']['id']
        element = get_first_exportable_element(doc['id'], workspace_id)
        if element:
            export_element(doc['id'], workspace_id, element)
        else:
            print("No exportable Part Studio or Assembly found in this document.")