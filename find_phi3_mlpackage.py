import requests

def search_hf_for_mlpackage():
    url = "https://huggingface.co/api/models?search=phi&filter=coreml"
    response = requests.get(url)
    if response.status_code == 200:
        for model in response.json():
            print(model['id'])
    else:
        print("Failed to fetch")

search_hf_for_mlpackage()
