from requests import post
from os import getenv
from dotenv import load_dotenv
load_dotenv()

apis = {
    "/task_1":"привет, может быть мы подружимся",
    "/task_2":"Больше не куплю"
}
for api,text in apis.items():
    result = post(
        getenv("DOMAIN")+api,
        json={
            "text":text
        }
    )

    print(f"{api}:",result.text)