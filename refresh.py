import requests
import secrets


class Refresh:

    def __init__(self):
        self.refresh_token=secrets.refresh_token
        self.base_64=secrets.base_64

    def refresh(self):
        query="https://accounts.spotify.com/api/token"
        response = requests.post(query,
        data={"grant_type":"refresh_token","refresh_token":secrets.refresh_token},
        headers={"Authorization":"Basic "+ secrets.base_64})
        return(response.json()["access_token"])

