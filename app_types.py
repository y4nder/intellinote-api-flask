from dataclasses import dataclass

@dataclass
class Keyword:
    keyword: str
    score: float

    def __str__(self):
        return f"Keyword: {self.keyword}, Score: {self.score}"


@dataclass    
class GeneratedResponse :
    keywords: list[Keyword]
    summary: str
    
    