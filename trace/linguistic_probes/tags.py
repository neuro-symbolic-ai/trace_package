from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class BaseTagSet(ABC):
    """
    Base class for tag sets with validation and common functionality.
    """
    tags: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize the tag set."""
        if not self.tags:
            self.tags = self._get_default_tags()

        self._validate_tags()

    @abstractmethod
    def _get_default_tags(self) -> Dict[str, int]:
        """Get default tag mappings (to be implemented by subclasses)."""
        pass

    def _validate_tags(self) -> None:
        """Validate tag mappings."""
        if not isinstance(self.tags, dict):
            raise TypeError("Tags must be a dictionary mapping tag names to indices.")

        if not all(isinstance(tag, str) for tag in self.tags.keys()):
            raise TypeError("All tag names must be strings.")

        if not all(isinstance(idx, int) for idx in self.tags.values()):
            raise TypeError("All tag indices must be integers.")

        if not all(idx >= 0 for idx in self.tags.values()):
            raise ValueError("All tag indices must be non-negative.")

        # Check for duplicate indices
        indices = list(self.tags.values())
        if len(indices) != len(set(indices)):
            raise ValueError("Tag indices must be unique.")

    def __getitem__(self, tag: str) -> int:
        """Get the index for a specific tag."""
        return self.tags.get(tag, self.tags.get("OTHER", -1))

    def __setitem__(self, tag: str, index: int) -> None:
        """Set the index for a specific tag."""
        if not isinstance(index, int) or index < 0:
            raise ValueError("Index must be a non-negative integer.")
        self.tags[tag] = index

    def __contains__(self, tag: str) -> bool:
        """Check if a tag exists in the set."""
        return tag in self.tags

    def __len__(self) -> int:
        """Get the number of tags."""
        return len(self.tags)

    def __iter__(self):
        """Iterate over tag names."""
        return iter(self.tags)

    def items(self):
        """Return items for compatibility."""
        return self.tags.items()

    def keys(self):
        """Return tag names."""
        return self.tags.keys()

    def values(self):
        """Return tag indices."""
        return self.tags.values()

    def get_tag_names(self) -> List[str]:
        """Get ordered list of tag names by index."""
        sorted_tags = sorted(self.tags.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_tags]

    def get_index(self, tag: str, default: Optional[int] = None) -> int:
        """Get index for a tag with optional default."""
        if default is None:
            default = self.tags.get("OTHER", -1)
        return self.tags.get(tag, default)

    def add_tag(self, tag: str, index: Optional[int] = None) -> None:
        """Add a new tag to the set."""
        if index is None:
            # Auto-assign next available index
            index = max(self.tags.values()) + 1 if self.tags else 0

        if index in self.tags.values():
            raise ValueError(f"Index {index} already exists for tag '{self._get_tag_by_index(index)}'")

        self.tags[tag] = index

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the set."""
        if tag not in self.tags:
            raise KeyError(f"Tag '{tag}' not found in tag set")
        del self.tags[tag]
        # reorder indices after removal
        self.tags = {name: idx for idx, (name, _) in enumerate(sorted(self.tags.items(), key=lambda x: x[1]))}

    def _get_tag_by_index(self, index: int) -> Optional[str]:
        """Get tag name by index."""
        for tag, idx in self.tags.items():
            if idx == index:
                return tag
        return None


@dataclass
class POSTagSet(BaseTagSet):
    """
    Part-of-Speech tag set with configurable granularity.
    """
    granularity: str = "basic"  # "basic" or "detailed"

    def __post_init__(self):
        """Initialize POS tag set with specified granularity."""
        if self.granularity not in ["basic", "detailed"]:
            raise ValueError("Granularity must be 'basic' or 'detailed'")
        super().__post_init__()

    def _get_default_tags(self) -> Dict[str, int]:
        """Get default POS tags based on granularity."""
        if self.granularity == "basic":
            return {
                "NOUN": 0,
                "VERB": 1,
                "ADJ": 2,
                "ADV": 3,
                "PREP": 4,
                "CONJ": 5,
                "OTHER": 6,
                # "DET": 5,
            }
        else:  # detailed
            return {
                "NOUN": 0,
                "TRANSITIVE_VERB": 1,
                "INTRANSITIVE_VERB": 2,
                "COMMUNICATION_VERB": 3,
                "MOTION_VERB": 4,
                "CHANGE_VERB": 5,
                "ADJ": 6,
                "ADV": 7,
                "LOCATION": 8,
                "TEMP": 9,
                "PREP": 10,
                "RESULT": 11,
                "CONJ": 12,
                "OTHER": 13
            }

    @classmethod
    def basic(cls, **kwargs) -> 'POSTagSet':
        """Create basic POS tag set."""
        return cls(granularity="basic", **kwargs)

    @classmethod
    def detailed(cls, **kwargs) -> 'POSTagSet':
        """Create detailed POS tag set."""
        return cls(granularity="detailed", **kwargs)


@dataclass
class SemanticTagSet(BaseTagSet):
    """
    Semantic role tag set with configurable granularity.
    """
    granularity: str = "basic"  # "basic" or "detailed"

    def __post_init__(self):
        """Initialize semantic tag set with specified granularity."""
        if self.granularity not in ["basic", "detailed"]:
            raise ValueError("Granularity must be 'basic' or 'detailed'")
        super().__post_init__()

    def _get_default_tags(self) -> Dict[str, int]:
        """Get default semantic tags based on granularity."""
        if self.granularity == "basic":
            return {
                "AGENT": 0,
                "PATIENT": 1,
                "ACTION": 2,
                "LOCATION": 3,
                "RELATION": 4,
                "CONNECTOR": 5,
                "RESULT": 6,
                "OTHER": 7
            }
        else:  # DETAILED
            return {
                "AGENT": 0,
                "PATIENT": 1,
                "ACTION": 2,
                "MOTION": 3,
                "COMMUNICATION": 4,
                "CHANGE": 5,
                "LOCATION": 6,
                "DESTINATION": 7,
                "TIME": 8,
                "RESULT": 9,
                "PROPERTY": 10,
                "MANNER": 11,
                "RELATION": 12,
                "CONNECTOR": 13,
                "OTHER": 14
            }

    @classmethod
    def basic(cls, **kwargs) -> 'SemanticTagSet':
        """Create basic semantic tag set."""
        return cls(granularity="basic", **kwargs)

    @classmethod
    def detailed(cls, **kwargs) -> 'SemanticTagSet':
        """Create DETAILED semantic tag set."""
        return cls(granularity="detailed", **kwargs)
