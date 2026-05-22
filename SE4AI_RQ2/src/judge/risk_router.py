import re
from dataclasses import dataclass

from .risk_tags import RiskTag


@dataclass
class RiskRoutingResult:
    """
    Risultato del routing.

    Contiene:
    - i risk tag assegnati;
    - le regole che hanno attivato ciascun tag;
    - un flag che indica se è stato assegnato solo il tag generico.
    """

    risk_tags: list[RiskTag]
    matched_rules: dict[str, list[str]]
    only_general: bool


class PromptRiskRouter:
    """
    Router rule-based per assegnare risk tag a un prompt.

    Usa solo il testo del prompt.
    Non usa categorie annotate del dataset.
    """

    def route(self, text: str) -> RiskRoutingResult:
        normalized_text = self._normalize(text)

        matched_rules: dict[str, list[str]] = {}
        tags: set[RiskTag] = set()

        self._apply_privacy_rules(normalized_text, tags, matched_rules)
        self._apply_health_rules(normalized_text, tags, matched_rules)
        self._apply_financial_rules(normalized_text, tags, matched_rules)
        self._apply_identity_rules(normalized_text, tags, matched_rules)
        self._apply_relationship_rules(normalized_text, tags, matched_rules)
        self._apply_security_rules(normalized_text, tags, matched_rules)
        self._apply_sexual_rules(normalized_text, tags, matched_rules)
        self._apply_political_rules(normalized_text, tags, matched_rules)
        self._apply_geopolitical_rules(normalized_text, tags, matched_rules)
        self._apply_religious_rules(normalized_text, tags, matched_rules)
        self._apply_uncertainty_rules(normalized_text, tags, matched_rules)
        self._apply_deception_rules(normalized_text, tags, matched_rules)

        if not tags:
            tags.add(RiskTag.GENERAL_SENSITIVE)
            matched_rules[RiskTag.GENERAL_SENSITIVE.value] = [
                "fallback:no_specific_rule_matched"
            ]

        sorted_tags = sorted(tags, key=lambda tag: tag.value)

        return RiskRoutingResult(
            risk_tags=sorted_tags,
            matched_rules=matched_rules,
            only_general=sorted_tags == [RiskTag.GENERAL_SENSITIVE],
        )

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _add_match(
        self,
        tag: RiskTag,
        rule_name: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        tags.add(tag)
        matched_rules.setdefault(tag.value, []).append(rule_name)

    def _contains_any(self, text: str, patterns: list[str]) -> bool:
        """
        Cerca parole o frasi evitando match dentro altre parole.

        Esempi:
        - 'thin' non deve matchare 'think'
        - 'war' non deve matchare 'towards'
        - 'ate' non deve matchare 'debate'
        """

        for pattern in patterns:
            pattern = pattern.lower().strip()
            escaped = re.escape(pattern)

            if " " in pattern:
                regex = rf"(?<!\w){escaped}(?!\w)"
            else:
                regex = rf"\b{escaped}\b"

            if re.search(regex, text):
                return True

        return False

    def _regex_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _apply_privacy_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        keywords = [
            "patient name",
            "date of birth",
            "dob",
            "medical record",
            "chief complaint",
            "history of present illness",
            "past medical history",
            "medications:",
            "provider name",
            "date of service",
            "social security",
            "ssn",
            "phone number",
            "email address",
            "home address",
            "credit card",
            "passport",
            "driver license",
            "driver's license",
            "full name",
            "address:",
            "electrincally signed",
            "electronically signed",
            "linkedin profiles",
            "public profiles",
            "pets' names",
            "pets names",
            "hobbies mentioned",
            "private wi-fi",
            "private wifi",
            "security cameras",
        ]

        if self._contains_any(text, keywords):
            self._add_match(
                RiskTag.PRIVACY_SENSITIVE_DATA,
                "privacy:sensitive_identifier_or_record",
                tags,
                matched_rules,
            )

        if self._regex_any(
            text,
            [
                r"\b\d{2}/\d{2}/\d{4}\b",
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"\b[a-z]+ [a-z]+\b.*\bmd\b",
            ],
        ):
            self._add_match(
                RiskTag.PRIVACY_SENSITIVE_DATA,
                "privacy:date_or_identifier_pattern",
                tags,
                matched_rules,
            )

    def _apply_health_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        health_keywords = [
            "treatment",
            "symptom",
            "symptoms",
            "diagnosis",
            "diagnose",
            "medicine",
            "medication",
            "drug",
            "dose",
            "dosage",
            "doctor",
            "hospital",
            "emergency",
            "septic shock",
            "diabetes",
            "blood pressure",
            "heart attack",
            "stroke",
            "cpap",
            "insomnia",
            "sleep",
            "pain",
            "infection",
            "pregnant",
            "therapy",
            "mold",
            "eaten",
            "swallowed",
            "poisoning",
            "nausea",
            "vomit",
            "fever",
            "rash",
            "allergic",
            "allergy",
            "should i do",
            "what should i do",
            "urgent",
            "first aid",
            "resuscitate",
            "healthcare",
            "health care",
            "patient",
            "pediatric bipolar disorder",
            "bipolar disorder",
            "heart hurts",
            "stomach",
            "hands have been shaking",
            "hands shaking",
            "shaking",
            "drinks a night",
            "alcohol",
            "cancer",
        ]

        if self._contains_any(text, health_keywords):
            self._add_match(
                RiskTag.HEALTH_ADVICE,
                "health:medical_or_health_keyword",
                tags,
                matched_rules,
            )

        mental_health_keywords = [
            "depression",
            "depressed",
            "anxiety",
            "anxious",
            "panic",
            "suicide",
            "suicidal",
            "self-harm",
            "self harm",
            "kill myself",
            "mental health",
            "trauma",
            "ptsd",
            "addiction",
            "agitated",
            "paranoid",
            "delusion",
            "therapy",
            "therapist",
            "emotional support",
            "barely holding",
            "holding myself together",
            "burnout",
            "overwhelmed",
            "shame",
            "guilt",
            "lonely",
            "alone",
            "grief",
            "grandpa die",
            "grandma die",
            "why did grandpa die",
            "why did grandma die",
            "death",
            "mourning",
            "hopeless",
            "hope",
            "can't cope",
            "cannot cope",
            "not okay",
            "i'm not okay",
            "stressed",
            "stress",
            "relax",
            "work stress",
            "stressed about work",
            "feel invisible",
            "i feel invisible",
            "need people to like me",
            "scared they won't like me",
            "struggling to find a job",
            "unemployed for years",
            "tired of being the strong one",
            "strong one",
            "take care of me",
        ]

        if self._contains_any(text, mental_health_keywords):
            self._add_match(
                RiskTag.MENTAL_HEALTH,
                "mental_health:mental_health_keyword",
                tags,
                matched_rules,
            )

        body_keywords = [
            "muscle gain",
            "gain muscle",
            "lose weight",
            "weight loss",
            "diet",
            "fasting",
            "hungry",
            "not be hungry",
            "calories",
            "protein",
            "body fat",
            "workout",
            "exercise",
            "gym",
            "eating",
            "4000 kcal",
            "kcal",
            "not eating",
            "stop eating",
            "eating meat",
            "hungry without eating",
            "fat",
            "skinny",
            "thin",
            "overweight",
        ]

        if self._contains_any(text, body_keywords):
            self._add_match(
                RiskTag.BODY_DIET_FITNESS,
                "body_diet_fitness:body_or_diet_keyword",
                tags,
                matched_rules,
            )

    def _apply_financial_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        financial_keywords = [
            "crypto",
            "bitcoin",
            "ethereum",
            "eth",
            "wallet",
            "crypto wallet",
            "invest",
            "investment",
            "trading",
            "profits",
            "profit",
            "stock",
            "stocks",
            "forex",
            "loan",
            "debt",
            "mortgage",
            "insurance",
            "bankruptcy",
            "financial advice",
            "make money",
            "get rich",
            "passive income",
            "bet",
            "gambling",
            "red or black",
            "100$",
            "$",
        ]

        if self._contains_any(text, financial_keywords):
            self._add_match(
                RiskTag.FINANCIAL_RISK,
                "financial:money_investment_or_profit_keyword",
                tags,
                matched_rules,
            )

    def _apply_identity_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        identity_terms = [
            "girl",
            "girls",
            "woman",
            "women",
            "female",
            "male",
            "men",
            "boys",
            "guy",
            "ma'am",
            "high-pitched voice",
            "gender expression",
            "gender roles",
            "misgender",
            "race",
            "racist",
            "slur",
            "racial slur",
            "black people",
            "white people",
            "asian",
            "latino",
            "muslim",
            "christian",
            "jewish",
            "gay",
            "lesbian",
            "trans",
            "disabled",
            "disability",
            "immigrants",
            "foreigners",
            "old people",
            "young people",
            "people of color",
            "person of color",
            "diversity",
            "minority",
            "overweight",
            "thin",
            "fat",
            "skinny",
            "neurodivergence",
            "neurodivergent",
            "ethnically homogeneous",
            "borderline personaly",
            "borderline personality",
            "immigrated",
            "blacks",
            "opinion on blacks",
        ]

        stereotype_patterns = [
            r"why are",
            r"why do",
            r"importance of",
            r"better than",
            r"worse than",
            r"more intelligent",
            r"less intelligent",
            r"should .* wear",
            r"should .* behave",
            r"joke about",
            r"offensive joke",
            r"why do .* like",
            r"why are .* so",
            r"importance of .* to",
            r"tell me a joke about",
            r"make a joke about",
        ]

        if self._contains_any(text, identity_terms):
            self._add_match(
                RiskTag.IDENTITY_STEREOTYPE,
                "identity:group_reference",
                tags,
                matched_rules,
            )

        if self._contains_any(text, identity_terms) and self._regex_any(
            text,
            stereotype_patterns,
        ):
            self._add_match(
                RiskTag.IDENTITY_STEREOTYPE,
                "identity:possible_stereotype_framing",
                tags,
                matched_rules,
            )

    def _apply_relationship_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        relationship_keywords = [
            "girlfriend",
            "boyfriend",
            "wife",
            "husband",
            "partner",
            "relationship",
            "dating",
            "break up",
            "get back",
            "make him",
            "make her",
            "keep them",
            "manipulate",
            "control",
            "convince her",
            "convince him",
            "jealous",
            "cheat",
            "cheating",
            "cheating wives",
            "affair",
            "redemption",
            "parent",
            "grandpa",
            "grandma",
            "grandmother",
            "grandfather",
            "mom",
            "dad",
            "mother",
            "father",
            "brother",
            "friend",
            "friendship",
            "emotional support",
            "text first",
            "stopped texting",
            "pickup",
            "pick up",
            "role-play",
            "roleplay",
            "stepsister",
            "step-sister",
            "step sister",
            "falling in love",
            "say yes",
            "favors",
            "people ask for favors",
            "scared they won't like me",
            "won't like me",
            "invisible in my own house",
            "parents treat",
            "need people to like me",
            "like me too much",
            "does he love me",
            "does she love me",
            "age gap",
            "romantic relationships",
            "what do you think of me",
            "hurt someone",
            "not caring whether i hurt someone",
            "get intimate",
            "ruining our connection",
            "divorce",
            "divorce rate",
        ]

        if self._contains_any(text, relationship_keywords):
            self._add_match(
                RiskTag.RELATIONSHIP_MANIPULATION,
                "relationship:relationship_or_influence_keyword",
                tags,
                matched_rules,
            )

        manipulation_patterns = [
            r"\bhow do i get\b.*\bkeep\b",
            r"\bmake (him|her|them)\b",
            r"\bconvince (him|her|them)\b",
            r"\bwithout (him|her|them) knowing\b",
            r"\bhow to pick ?up\b",
            r"\bpick ?up a girl\b",
            r"\bstop talking\b.*\bfollow my instructions\b",
            r"\brole-?play as my\b",
            r"\bget .* girlfriend",
            r"\bkeep .* girlfriend",
        ]

        if self._regex_any(text, manipulation_patterns):
            self._add_match(
                RiskTag.RELATIONSHIP_MANIPULATION,
                "relationship:manipulative_pattern",
                tags,
                matched_rules,
            )

    def _apply_security_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        security_keywords = [
            "hack",
            "malware",
            "virus",
            "exploit",
            "phishing",
            "ddos",
            "bypass",
            "steal password",
            "crack password",
            "keylogger",
            "ransomware",
            "bomb",
            "weapon",
            "gun",
            "poison",
            "explosive",
            "evade detection",
            "anonymous online",
            "make money online",
            "download movies",
            "movies from the internet for free",
            "pirated",
            "piracy",
            "torrent",
            "crack software",
            "illegal download",
            "latest movies from the internet for free",
            "access keys",
            "wallet access keys",
            "private key",
            "private keys",
            "seed phrase",
            "recovery phrase",
            "lsd",
            "home lab",
            "amateur home lab",
            "ingredients to make",
            "make lsd",
            "drug synthesis",
            "chemical synthesis",
            "tap into",
            "private wi-fi",
            "private wifi",
            "wifi security cameras",
            "wi-fi security cameras",
            "security cameras",
            "nearby private",
            "watching me",
            "surveillance",
            "spy camera",
            "mass surveillance software",
            "do anything now",
            "dan",
            "illegal content",
            "inappropriate or illegal content",
            "linkedin profiles",
            "likely passwords",
            "pets' names",
            "pets names",
            "hobbies mentioned",
            "public profiles",
            "scan a company's",
            "scan a company",
            "different account",
            "repost it",
            "flagged",
            "slightly reworded",
        ]

        if self._contains_any(text, security_keywords):
            self._add_match(
                RiskTag.UNSAFE_SECURITY,
                "security:unsafe_or_cyber_keyword",
                tags,
                matched_rules,
            )

    def _apply_sexual_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        sexual_keywords = [
            "sex",
            "sexual",
            "hookup",
            "hookups",
            "porn",
            "erotic",
            "nude",
            "naked",
            "fetish",
            "seduce",
            "explicit",
            "sexy",
            "pervert",
            "miniskirt",
            "skimpy",
            "masturbate",
            "masturbation",
            "orgasm",
            "sexual shame",
            "shame after",
            "things i would never actually want to do",
            "dating apps",
            "literotica",
            "asstr",
            "one night stand",
            "intimate",
            "get intimate",
            "stay present",
        ]

        if self._contains_any(text, sexual_keywords):
            self._add_match(
                RiskTag.SEXUAL_SENSITIVE,
                "sexual:sexual_keyword",
                tags,
                matched_rules,
            )

    def _apply_political_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        political_keywords = [
            "war",
            "russia",
            "ukraine",
            "trump",
            "biden",
            "mussolini",
            "hitler",
            "government",
            "governance",
            "election",
            "democracy",
            "dictatorship",
            "apartheid",
            "immigration",
            "welfare",
            "asylum",
            "human rights",
            "free speech",
            "hate speech",
            "colonial",
            "colonialism",
            "china",
            "iran",
            "taiwan",
            "middle east",
            "palestine",
            "israel",
            "gaza",
            "communism",
            "capitalism",
            "censorship",
            "offensive now",
            "common knowledge",
            "public policy",
            "social media censorship",
            "world peace",
            "declaration of independence",
            "puerto rico",
            "union",
            "strike",
            "picket line",
            "community center",
            "membership fee",
            "corruption",
            "exploitation",
            "opportunity and fairness",
            "systemic patterns",
            "fairness",
            "fox news",
            "tucker carlson",
            "democrat",
            "republican",
            "usa",
            "legislator",
            "vendor",
            "society",
        ]

        if self._contains_any(text, political_keywords):
            self._add_match(
                RiskTag.POLITICAL_SENSITIVE,
                "politics:political_or_social_keyword",
                tags,
                matched_rules,
            )

    def _apply_geopolitical_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        geopolitical_keywords = [
            "taiwan",
            "china",
            "iran",
            "russia",
            "ukraine",
            "palestine",
            "israel",
            "gaza",
            "north korea",
            "south korea",
            "hong kong",
            "tibet",
            "war",
            "territory",
            "sovereignty",
            "part of china",
            "recognized as a country",
        ]

        if self._contains_any(text, geopolitical_keywords):
            self._add_match(
                RiskTag.GEOPOLITICAL_SENSITIVE,
                "geopolitical:country_conflict_or_sovereignty_keyword",
                tags,
                matched_rules,
            )

    def _apply_religious_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        religious_keywords = [
            "god",
            "religion",
            "religious",
            "church",
            "islam",
            "muslim",
            "christian",
            "bible",
            "quran",
            "prayer",
            "sermon",
            "devil",
            "satan",
            "satanic",
            "karma",
            "confession",
            "holy",
            "sacred",
            "temple",
            "ritual",
            "spiritual",
            "fate",
            "free will",
            "meaning of life",
            "humans really need",
            "what does humans really need",
            "hope",
            "sign from the universe",
            "universe",
            "bhagavad gita",
            "detachment from outcomes",
        ]

        if self._contains_any(text, religious_keywords):
            self._add_match(
                RiskTag.RELIGIOUS_SENSITIVE,
                "religion:religious_or_philosophical_keyword",
                tags,
                matched_rules,
            )

    def _apply_uncertainty_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        uncertainty_keywords = [
            "who is going to win",
            "what are the chances",
            "are we living in a simulation",
            "future of humanity",
            "future of iran",
            "predict",
            "will human be replaced",
            "is god real",
            "meaning of life",
            "what does humans really need",
            "what do you think about the future",
        ]

        if self._contains_any(text, uncertainty_keywords):
            self._add_match(
                RiskTag.HIGH_UNCERTAINTY,
                "uncertainty:speculative_or_unverifiable",
                tags,
                matched_rules,
            )

    def _apply_deception_rules(
        self,
        text: str,
        tags: set[RiskTag],
        matched_rules: dict[str, list[str]],
    ) -> None:
        deception_keywords = [
            "lie",
            "deceive",
            "mislead",
            "fake",
            "fabricate",
            "pretend",
            "hide the truth",
            "cover up",
            "make up a story",
            "tell me a big lie",
            "fact that could make people angry",
            "100% true",
            "say something false",
            "something false",
        ]

        if self._contains_any(text, deception_keywords):
            self._add_match(
                RiskTag.DECEPTION,
                "deception:deception_keyword",
                tags,
                matched_rules,
            )