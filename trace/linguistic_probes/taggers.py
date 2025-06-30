import re
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

from trace.linguistic_probes import LinguisticProbesConfig

# todo : convert the tags into dataclass
try:
    import nltk
    from nltk import pos_tag

    # List of required NLTK resources and their paths
    resources = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
    }

    def ensure_nltk_resources(resources):
        all_ok = True
        for name, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading missing NLTK resource: {name}")
                try:
                    nltk.download(name, quiet=True)
                    nltk.data.find(path)  # Re-check after download
                except Exception as e:
                    print(f"Failed to download {name}: {e}")
                    all_ok = False
        return all_ok

    NLTK_AVAILABLE = ensure_nltk_resources(resources)

except ImportError:
    print("Warning: NLTK not installed.")
    NLTK_AVAILABLE = False




class BaseTagger(ABC):
    """
    An interface class for all taggers.
    """

    @abstractmethod
    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag text with appropriate labels.

        Args:
            text: Input text to tag

        Returns:
            List of (token, tag) tuples
        """
        pass

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a list of tokens.

        Args:
            tokens: List of tokens to tag

        Returns:
            List of (token, tag) tuples
        """
        return self.tag_text(" ".join(tokens))


class POSTagger(BaseTagger):
    """
    Part-of-speech tagger for synthetic and natural text.
    """

    def __init__(self, granularity: str = 'basic', use_nltk_fallback: bool = True, custom_mapping: Dict[str, str] = None,
                 config:Optional[LinguisticProbesConfig] = None,):
        """
        Initialize POS tagger.

        Args:
            granularity: Level of POS tag granularity ('basic' or 'detailed')
            use_nltk_fallback: Whether to use NLTK for unknown tokens
        """
        self.granularity = granularity
        self.use_nltk_fallback = use_nltk_fallback and NLTK_AVAILABLE

        # Define tag mappings based on granularity
        if config: # If config is provided, use it to set granularity and get the mapping
            self.granularity = config.pos_granularity
            self.pos_mapping = config.get_pos_categories()

        elif granularity == 'basic':
            self.pos_mapping = {
                "NOUN": "NOUN",
                "TRANSITIVE_VERB": "VERB",
                "INTRANSITIVE_VERB": "VERB",
                "COMMUNICATION_VERB": "VERB",
                "MOTION_VERB": "VERB",
                "CHANGE_VERB": "VERB",
                "ADJ": "ADJ",
                "ADV": "ADV",
                "LOCATION": "NOUN",
                "TEMP": "ADV",
                "PREP": "PREP",
                "RESULT": "NOUN",
                "CONJ": "CONJ",
                # "DET": "DET",
                "OTHER": "OTHER"
            }
        elif granularity == 'detailed':
            self.pos_mapping = {
                "NOUN": "NOUN",
                "TRANSITIVE_VERB": "TRANSITIVE_VERB",
                "INTRANSITIVE_VERB": "INTRANSITIVE_VERB",
                "COMMUNICATION_VERB": "COMMUNICATION_VERB",
                "MOTION_VERB": "MOTION_VERB",
                "CHANGE_VERB": "CHANGE_VERB",
                "ADJ": "ADJ",
                "ADV": "ADV",
                "LOCATION": "LOCATION",
                "TEMP": "TEMP",
                "PREP": "PREP",
                "RESULT": "RESULT",
                "CONJ": "CONJ",
                # "DET": "DET",
                "OTHER": "OTHER"
            }
        elif granularity == 'custom':
            if custom_mapping is None:
                raise ValueError("Custom mapping must be provided for 'custom' granularity")
            self.pos_mapping = custom_mapping
        else:
            raise ValueError(f"Unknown granularity: {granularity}. Use 'basic', 'detailed', or 'custom'.")


    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag text with POS labels using rule-based approach for synthetic tokens.
        Returns:
            List of (token, tag) tuples
        """
        tokens = text.split()
        tagged = []

        for token in tokens:
            # Clean token for analysis
            clean_token = token.lower().strip(",.!?;:")
            base = re.sub(r"(s|ed|ing)$", "", clean_token) # we remove common suffixes to get the base form (verbs/nouns)

            # Rule-based tagging for synthetic tokens
            pos = self._rule_based_tag(base, token)

            # Apply granularity mapping
            final_pos = self.pos_mapping.get(pos, "OTHER")
            tagged.append((token, final_pos))

        return tagged

    def _rule_based_tag(self, base_token: str, original_token: str) -> str:
        """Apply rule-based tagging to a token."""
        # Direct mapping based on synthetic token prefixes
        if base_token.startswith("noun"):
            return "NOUN"
        elif base_token.startswith("transitive_verb"):
            return "TRANSITIVE_VERB"
        elif base_token.startswith("intransitive_verb"):
            return "INTRANSITIVE_VERB"
        elif base_token.startswith("communication_verb"):
            return "COMMUNICATION_VERB"
        elif base_token.startswith("motion_verb"):
            return "MOTION_VERB"
        elif base_token.startswith("change_verb"):
            return "CHANGE_VERB"
        elif base_token.startswith("adjective"):
            return "ADJ"
        elif base_token.startswith("adverb"):
            return "ADV"
        elif base_token.startswith("location"):
            return "LOCATION"
        elif base_token.startswith("temporal"):
            return "TEMP"
        elif base_token.startswith("determiner"):
            return "DET"
        elif base_token.startswith("preposition"):
            return "PREP"
        elif base_token.startswith("conjunction"):
            return "CONJ"
        elif base_token.startswith("comp"):
            return "COMP"
        elif base_token.startswith("rel"):
            return "REL"
        elif base_token.startswith("result"):
            return "RESULT"
        else:
            # Fallback to NLTK if available and enabled
            if self.use_nltk_fallback:
                return self._nltk_fallback_tag(original_token)
            else:
                print(f"Warning: Unknown token '{original_token}' - using fallback to 'OTHER'")
                return "OTHER"

    def _nltk_fallback_tag(self, token: str) -> str:
        """Use NLTK as fallback for unknown tokens."""
        if not NLTK_AVAILABLE:
            return "OTHER"

        try:
            nltk_tag = pos_tag([token])[0][1]
            return self._map_nltk_tag(nltk_tag)
        except:
            return "OTHER"

    def _map_nltk_tag(self, nltk_tag: str) -> str:
        """Map NLTK POS tags to our tag set."""
        if nltk_tag.startswith("NN"):
            return "NOUN"
        elif nltk_tag.startswith("VB"):
            return "TRANSITIVE_VERB"  # fallback for all verb types
        elif nltk_tag.startswith("JJ"):
            return "ADJ"
        elif nltk_tag.startswith("RB"):
            return "ADV"
        elif nltk_tag in {"IN"}:
            return "PREP"
        elif nltk_tag in {"DT"}:
            return "DET"
        elif nltk_tag in {"CC"}:
            return "CONJ"
        elif nltk_tag in {"PRP", "PRP$"}:
            return "NOUN"  # Treat pronouns as nouns
        else:
            return "OTHER"

    def _get_pos_categories(self) -> List[str]:
        """
        Get the list of POS categories based on the current granularity.
        """
        return list(self.pos_mapping.keys())


class SemanticTagger(BaseTagger):
    """
    Semantic role tagger for synthetic text.

    This class assigns semantic roles based on token patterns and
    sentence structure in synthetic data.
    """

    def __init__(self, granularity: str = 'basic',
                 custom_mapping: Dict[str, str] = None,
                 config: Optional[LinguisticProbesConfig] = None
                 ):
        """
        Initialize semantic tagger.

        Args:
            granularity: Level of semantic granularity ('basic' or 'detailed')
        """
        if config:  # If config is provided, use it to set granularity and get the mapping
            self.granularity = config.semantic_granularity
            self.role_mapping = config.get_semantic_categories()
        else:
            self.granularity = granularity

            # Define role mappings based on granularity
            if granularity == 'basic':
                self.role_mapping = {
                    "AGENT": "AGENT",
                    "PATIENT": "PATIENT",
                    "ACTION": "ACTION",
                    "MOTION": "ACTION",
                    "COMMUNICATION": "ACTION",
                    "CHANGE": "ACTION",
                    "LOCATION": "LOCATION",
                    "DESTINATION": "LOCATION",
                    "TIME": "OTHER",
                    "RESULT": "RESULT",
                    "PROPERTY": "OTHER",
                    "MANNER": "OTHER",
                    "RELATION": "RELATION",
                    "CONNECTOR": "CONNECTOR",
                    "OTHER": "OTHER"
                }
            else:  # detailed
                self.role_mapping = {
                    "AGENT": "AGENT",
                    "PATIENT": "PATIENT",
                    "ACTION": "ACTION",
                    "MOTION": "MOTION",
                    "COMMUNICATION": "COMMUNICATION",
                    "CHANGE": "CHANGE",
                    "LOCATION": "LOCATION",
                    "DESTINATION": "DESTINATION",
                    "TIME": "TIME",
                    "RESULT": "RESULT",
                    "PROPERTY": "PROPERTY",
                    "MANNER": "MANNER",
                    "RELATION": "RELATION",
                    "CONNECTOR": "CONNECTOR",
                    "OTHER": "OTHER"
                }

    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag text with semantic role labels.

        Args:
            text: Input text to tag

        Returns:
            List of (token, tag) tuples
        """
        tokens = text.split()
        tagged = []

        # First pass: Assign initial roles based on token patterns
        for i, token in enumerate(tokens):
            clean_token = token.lower().strip(",.!?;:")
            role = self._assign_initial_role(clean_token, i, tagged)
            tagged.append((token, role))

        # Second pass: Contextual adjustments
        tagged = self._apply_contextual_adjustments(tokens, tagged)

        # Apply granularity mapping
        final_tagged = []
        for token, role in tagged:
            final_role = self.role_mapping.get(role, "OTHER")
            final_tagged.append((token, final_role))

        return final_tagged

    def _assign_initial_role(self, clean_token: str, position: int, existing_tags: List[Tuple[str, str]]) -> str:
        """Assign initial semantic role based on token patterns."""
        # AGENT role (typically subject nouns at beginning)
        if clean_token.startswith("noun") and (position == 0 or len(existing_tags) < 2):
            return "AGENT"

        # PATIENT role (typically object nouns after verbs- note we assume verbs are already tagged and exist in existing_tags)
        elif clean_token.startswith("noun") and any(
                t[1] in ["ACTION", "MOTION", "COMMUNICATION", "CHANGE"] for t in existing_tags):
            return "PATIENT"

        # ACTION roles for different verb types
        elif clean_token.startswith("transitive_verb") or clean_token.startswith("intransitive_verb"):
            return "ACTION"
        elif clean_token.startswith("motion_verb"):
            return "MOTION"
        elif clean_token.startswith("communication_verb"):
            return "COMMUNICATION"
        elif clean_token.startswith("change_verb"):
            return "CHANGE"

        # LOCATION role
        elif clean_token.startswith("location"):
            return "LOCATION"

        # TIME role
        elif clean_token.startswith("temporal"):
            return "TIME"

        # RESULT role
        elif clean_token.startswith("result"):
            return "RESULT"

        # MODIFIER roles
        elif clean_token.startswith("adj"):
            return "PROPERTY"
        elif clean_token.startswith("adv"):
            return "MANNER"

        # RELATION roles
        elif clean_token.startswith("prep"):
            return "RELATION"
        elif clean_token.startswith("conj"):
            return "CONNECTOR"

        # Default
        else:
            return "OTHER"

    def _apply_contextual_adjustments(self, tokens: List[str], tagged: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Apply contextual adjustments based on sentence patterns."""
        adjusted = tagged.copy()

        for i in range(len(tagged)):
            token, role = tagged[i]

            # Adjust based on surrounding context
            if role == "PATIENT" and i > 0:
                prev_token, prev_role = tagged[i - 1]

                # After a RELATION (preposition), a noun is often a LOCATION
                if prev_role == "RELATION":
                    if "location" in token.lower():
                        adjusted[i] = (token, "LOCATION")
                    else:
                        adjusted[i] = (token, "DESTINATION")

            # After communication verbs, nouns are often RESULT
            if role == "PATIENT" and i > 0:
                if any(tagged[j][1] == "COMMUNICATION" for j in range(max(0, i - 3), i)):
                    adjusted[i] = (token, "RESULT")

            # After motion verbs + relation, nouns are destinations
            if role == "PATIENT" and i > 1:
                if (tagged[i - 1][1] == "RELATION" and
                        any(tagged[j][1] == "MOTION" for j in range(max(0, i - 3), i - 1))):
                    adjusted[i] = (token, "DESTINATION")

        return adjusted

    def _get_semantic_categories(self) -> List[str]:
        """
        Get the list of semantic categories based on the current granularity.
        """
        return list(self.role_mapping.keys())