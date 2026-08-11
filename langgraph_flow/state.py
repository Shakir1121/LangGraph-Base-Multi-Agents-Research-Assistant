from typing import TypedDict


class ResearchState(TypedDict):

    query: str

    sections: dict

    vectorstore: object

    route: str

    response: str
