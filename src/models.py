class Stem:
    """
    Represents a musical stem with metadata and audio file information.

    A stem is a component of a musical composition that contains a specific
    instrument or group of instruments. This class stores both the metadata
    about the stem and references to the actual audio files.

    Attributes:
        id (str): Unique identifier for the stem
        mix_id (str | None): Identifier for the mix this stem belongs to
        dialog_uuid (str): UUID of the dialog session this stem was created in
        is_original (bool): Whether this is an original stem or generated
        is_block (bool): Whether this stem is a block (building block for composition)
        category (str): Category/type of the stem (e.g., 'drums', 'bass', 'melody')
        caption (str): Text description of the stem
        section_name (str): Name of the musical section this stem belongs to
        section_role (str): Role of this stem within the section
        bar_count (int): Number of musical bars in this stem
        bpm (int): Beats per minute (tempo) of the stem
        key (str): Musical key of the stem
        uri (str): S3 URI pointing to the audio file
        url (str | None): Presigned URL for direct access to the audio file
    """

    id: str
    mix_id: str | None
    dialog_uuid: str  # dialog uuid that this stem is generated
    is_original: bool
    is_block: bool
    category: str
    caption: str
    instrument_name: str
    section_name: str
    section_role: str
    bar_count: int
    bpm: int
    key: str
    uri: str  # S3 URI
    url: str | None  # Presigned URL
    used_block_ids: list[str]

    def __init__(
        self,
        id,
        mix_id,
        dialog_uuid,
        is_original,
        is_block,
        category,
        caption,
        instrument_name,
        section_name,
        section_role,
        bar_count,
        bpm,
        key,
        uri,
        url=None,
        used_block_ids=None,
    ):
        """
        Initialize a new Stem instance.

        Args:
            id (str): Unique identifier for the stem
            mix_id (str | None): Identifier for the mix this stem belongs to
            dialog_uuid (str): UUID of the dialog session this stem was created in
            is_original (bool): Whether this is an original stem or generated
            is_block (bool): Whether this stem is a block (building block for composition)
            category (str): Category/type of the stem
            caption (str): Text description of the stem
            section_name (str): Name of the musical section this stem belongs to
            section_role (str): Role of this stem within the section
            bar_count (int): Number of musical bars in this stem
            bpm (int): Beats per minute (tempo) of the stem
            key (str): Musical key of the stem
            uri (str): S3 URI pointing to the audio file
            url (str | None, optional): Presigned URL for direct access. Defaults to None.
        """
        self.id = id
        self.mix_id = mix_id
        self.dialog_uuid = dialog_uuid
        self.is_original = is_original
        self.is_block = is_block
        self.category = category
        self.caption = caption
        self.instrument_name = instrument_name
        self.section_name = section_name
        self.section_role = section_role
        self.bar_count = bar_count
        self.bpm = bpm
        self.key = key
        self.uri = uri
        self.url = url
        self.used_block_ids = used_block_ids if used_block_ids else []

    def to_dict(self):
        return {
            "id": self.id,
            "mixId": self.mix_id,
            "dialogUuid": self.dialog_uuid,
            "isOriginal": self.is_original,
            "isBlock": self.is_block,
            "category": self.category,
            "caption": self.caption,
            "instrumentName": self.instrument_name,
            "sectionName": self.section_name,
            "sectionRole": self.section_role,
            "barCount": self.bar_count,
            "bpm": self.bpm,
            "key": self.key,
            "uri": self.uri,
            "url": self.url,
            "usedBlockIds": self.used_block_ids,
        }


class ContextSong:
    """
    Represents contextual information about a song being worked on.

    This class stores metadata about the current song context, including
    musical properties and references to audio files that provide context
    for the current composition session.

    Attributes:
        song_id (str): Unique identifier for the song
        bpm (int): Beats per minute (tempo) of the song
        key (str): Musical key of the song
        bar_count (int): Number of musical bars in the song
        section_name (str): Name of the current musical section
        section_role (str): Role of the current section within the song
        context_audio_uris (list[str]): List of S3 URIs for context audio files
        created_sections_order (list[str]): Ordered list of section names as they were created
    """

    song_id: str
    bpm: int
    key: str
    bar_count: int
    section_name: str
    section_role: str
    song_structure: str
    context_audio_uris: list[str]  # S3 URIs
    created_sections_order: list[str]
    arranged_sections_order: list[str]
    is_remix: bool

    def __init__(
        self,
        song_id,
        bpm,
        key,
        bar_count,
        section_name,
        song_structure,
        section_role,
        context_audio_uris,
        created_sections_order,
        arranged_sections_order,
        is_remix,
    ):
        """
        Initialize a new ContextSong instance.

        Args:
            song_id (str): Unique identifier for the song
            bpm (int): Beats per minute (tempo) of the song
            key (str): Musical key of the song
            bar_count (int): Number of musical bars in the song
            section_name (str): Name of the current musical section
            section_role (str): Role of the current section within the song
            context_audio_uris (list[str]): List of S3 URIs for context audio files
            created_sections_order (list[str]): Ordered list of section names as they were created
            arranged_sections_order (list[str]): Ordered list of section names as they were arranged
            is_remix (bool): Whether the song is a remix
        """
        self.song_id = song_id
        self.bpm = bpm
        self.key = key
        self.bar_count = bar_count
        self.section_name = section_name
        self.song_structure = song_structure
        self.section_role = section_role
        self.context_audio_uris = context_audio_uris
        self.created_sections_order = created_sections_order
        self.arranged_sections_order = arranged_sections_order
        self.is_remix = is_remix

    def to_dict(self):
        """
        Convert the ContextSong instance to a dictionary representation.

        Returns:
            dict: Dictionary containing the song's metadata including audio URIs
        """

        # omit context_audio_uris
        return {
            "song_id": self.song_id,
            "bpm": self.bpm,
            "key": self.key,
            "bar_count": self.bar_count,
            "section_name": self.section_name,
            "song_structure": self.song_structure,
            "section_role": self.section_role,
            "context_audio_uris": self.context_audio_uris,
            "created_sections_order": self.created_sections_order,
            "arranged_sections_order": self.arranged_sections_order,
            "is_remix": self.is_remix,
        }

    def to_camel_case_dict(self):
        return {
            "songId": self.song_id,
            "bpm": self.bpm,
            "key": self.key,
            "barCount": self.bar_count,
            "sectionName": self.section_name,
            "sectionRole": self.section_role,
            "songStructure": self.song_structure,
            "contextAudioUris": self.context_audio_uris,
            "createdSectionsOrder": self.created_sections_order,
            "arrangedSectionsOrder": self.arranged_sections_order,
            "isRemix": self.is_remix,
        }


class MemoryData:
    """
    Represents the complete memory state for a music generation session.

    This class encapsulates all the data needed to maintain context and state
    during an interactive music generation session, including generated stems,
    conversation history, and current session information.

    Attributes:
        generated_stems (list[Stem]): List of all stems generated in this session
        previous_context (list[str]): Previous conversation context/messages
        stems_in_mix (list[Stem]): Stems currently included in the active mix
        context_song (ContextSong): Information about the current song context
        turn_index (int): Current turn number in the conversation
        intent_focused_prompt (str): Current focused prompt based on user intent
        mix_stem_diff (list[Stem]): List of stems in the current mix difference
        current_mix_stem_diff (Stem): Current stem being processed in the mix
    """

    memory_id: str  # file path or dialog_uuid
    user_prompt: str
    intent_focused_prompt: str
    intent_history: list[str]
    chosen_sections: list[list[Stem]]
    generated_stems: list[Stem]
    context_song: ContextSong
    turn_index: int
    working_section: list[Stem]  # try removing
    working_section_index: int

    def __init__(
        self,
        memory_id=None,
        user_prompt=None,
        intent_focused_prompt=None,
        intent_history=None,
        chosen_sections=None,
        generated_stems=None,
        context_song=None,
        turn_index=None,
        working_section=None,
        working_section_index=None,
    ):
        self.memory_id = memory_id
        self.user_prompt = user_prompt if user_prompt else ""
        self.intent_focused_prompt = (
            intent_focused_prompt if intent_focused_prompt else ""
        )
        self.intent_history = intent_history if intent_history else []
        self.chosen_sections = chosen_sections if chosen_sections else []
        self.generated_stems = generated_stems if generated_stems else []
        self.context_song = context_song if context_song else None
        self.turn_index = turn_index if turn_index else 1
        self.working_section_index = (
            working_section_index if working_section_index else 0
        )
        self.working_section = (
            working_section
            if working_section
            else (
                chosen_sections[self.working_section_index]
                if chosen_sections and self.working_section_index < len(chosen_sections)
                else []
            )
        )

    def print(self):
        print("MemoryData:")
        print(f"memory_id: {self.memory_id}")
        print(f"user_prompt: {self.user_prompt}")
        print(f"intent_focused_prompt: {self.intent_focused_prompt}")
        print(f"intent_history: {self.intent_history}")
        print(f"chosen_sections: {self.chosen_sections}")
        print(f"generated_stems: {[stem.to_dict() for stem in self.generated_stems]}")
        if self.context_song:
            print(f"context_song: {self.context_song.to_dict()}")
        print(f"turn_index: {self.turn_index}")
        print(f"working_section: {self.working_section}")
        print(f"working_section_index: {self.working_section_index}")


class MemoryStrategy:
    """
    Defines the strategy for how memory should be handled in a music generation session.

    This class encapsulates various boolean flags that determine how the system
    should approach memory management and stem generation based on user intent
    and session state.

    Attributes:
        should_load_older_memories (bool): Whether to load previous session memory
        is_start_from_scratch (bool): Whether to start a completely new session
        is_use_suggested_stems (bool): Whether to use suggested stems from previous sessions
        is_start_new_branch (bool): Whether to start a new branch of composition
        is_start_new_section (bool): Whether to start a new musical section
        target_category (str): Target category for the next stem generation
        intent_focused_prompt (str): User's expressed intent for the current interaction
        target_working_section_index (int | None): Index of the section to work on, or None if not specified
    """

    should_load_older_memories: bool
    is_start_from_scratch: bool
    is_use_suggested_stems: bool
    is_start_new_branch: bool
    is_start_new_section: bool
    is_publish_song: bool
    # target_category: str  # TODO: check if this is needed
    intent_focused_prompt: str
    selected_older_memory_ids: list[str]
    target_working_section_index: int | None

    def __init__(
        self,
        should_load_older_memories=False,
        is_start_from_scratch=True,
        is_use_suggested_stems=False,
        is_start_new_branch=False,
        is_start_new_section=False,
        is_publish_song=False,
        # target_category="",
        intent_focused_prompt="",
        selected_older_memory_ids=None,
        target_working_section_index=None,
    ):
        """
        Initialize a new MemoryStrategy instance.

        Args:
            should_load_older_memories (bool): Whether to load previous session memory
            is_start_from_scratch (bool): Whether to start a completely new session
            is_use_suggested_stems (bool): Whether to use suggested stems from previous sessions
            is_start_new_branch (bool): Whether to start a new branch of composition
            is_start_new_section (bool): Whether to start a new musical section
            target_category (str): Target category for the next stem generation
            intent_focused_prompt (str): User's expressed intent for the current interaction
            selected_older_memory_ids (list[str]): List of selected older memory ids
            target_working_section_index (int | None): Index of the section to work on, or None if not specified
        """
        self.should_load_older_memories = should_load_older_memories
        self.is_start_from_scratch = is_start_from_scratch
        self.is_use_suggested_stems = is_use_suggested_stems
        self.is_start_new_branch = is_start_new_branch
        self.is_start_new_section = is_start_new_section
        self.is_publish_song = is_publish_song
        # self.target_category = target_category
        self.intent_focused_prompt = intent_focused_prompt
        self.selected_older_memory_ids = (
            selected_older_memory_ids if selected_older_memory_ids else []
        )
        self.target_working_section_index = target_working_section_index

    def __str__(self):
        """Return a readable string representation of the MemoryStrategy."""
        return (
            f"MemoryStrategy(\n"
            f"  should_load_older_memories: {self.should_load_older_memories}\n"
            f"  start_from_scratch: {self.is_start_from_scratch}\n"
            f"  use_suggested_stems: {self.is_use_suggested_stems}\n"
            f"  start_new_branch: {self.is_start_new_branch}\n"
            f"  start_new_section: {self.is_start_new_section}\n"
            f"  publish_song: {self.is_publish_song}\n"
            # f"  target_category: '{self.target_category}'\n"
            f"  intent_focused_prompt: '{self.intent_focused_prompt}'\n"
            f"  selected_older_memory_ids: {self.selected_older_memory_ids}\n"
            f"  target_working_section_index: {self.target_working_section_index}\n"
            f")"
        )


class MemorySelection:
    """
    Represents the selection of memory data for a music generation session.

    This class encapsulates information about which memory data should be
    loaded or used for the current session, along with the strategy for
    how to handle that memory.

    Attributes:
        recent_memory_id (str): ID of the most recent memory entry
        recent_memory (MemoryData): The most recent memory data object
        selected_memory_ids (list[str]): List of selected memory entry IDs
        strategy (MemoryStrategy): Strategy for handling the selected memory

    """

    recent_memory_id: str
    recent_memory: MemoryData
    selected_memory_ids: list[str]
    strategy: MemoryStrategy
    # user_intent: str

    def __init__(
        self,
        recent_memory_id: str,
        recent_memory: MemoryData,
        selected_memory_ids: list[str],
        strategy: MemoryStrategy,
    ):
        """
        Initialize a new MemorySelection instance.

        Args:
            recent_memory_id (str): ID of the most recent memory entry
            recent_memory (MemoryData): The most recent memory data object
            selected_memory_ids (list[str]): List of selected memory entry IDs
            strategy (MemoryStrategy): Strategy for handling the selected memory

        """
        self.recent_memory_id = recent_memory_id
        self.recent_memory = recent_memory
        self.selected_memory_ids = selected_memory_ids
        self.strategy = strategy


class GenerateStemDiffOutput:
    """Generate stem diff tool에 대한 출력 스키마"""

    output_uris: list[str]

    def __init__(self, output_uris=None):
        self.output_uris = output_uris if output_uris else []

    def to_dict(self):
        return {
            "output_uris": self.output_uris,
        }
