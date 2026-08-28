# To use FAISS we need to install numpy, faiss-cpu.
# Note for the procedure
# Each new embedding needs to be indexed, and this can be saved locally to save time
# Each index is associated with unique embedding with a unique ID
# The typical steps are:
# 1. All chunks embedding must have a unique ID, and each ID will have the index, which can be precomputed.
#    The index could be large, each 768-dim embedding is 0.0294MB.
# 2. The index can be saved to disk and loaded when needed.
# 3. During search, we can get a subset of the embeddings and conduct the search.
# Some more benchmark data
# On my M1 Mac, the search time for 20k subset is 0.06s, not a problem
# For q query and n documents, without index, the complexity is O(q*n), with index, the complexity is O(q*log(n))

import os
import numpy as np
import faiss
import time
from multiprocessing.synchronize import Lock
from functools import wraps
from typing import Optional, List
from logging import getLogger

logger = getLogger(__name__)


def with_faiss_lock(write_operation=True):
    """Decorator to handle locking and index loading for write operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if write_operation and self.lock is not None:
                with self.lock:
                    # Load latest index from disk for write operations
                    if not self._load():
                        logger.info(f"Could not load index for {func.__name__}. Will create new one if needed.")
                    # Execute the function with the lock held
                    result = func(self, *args, **kwargs)
                    return result
            else:
                # No locking for read operations, simply return the original function
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class FaissIndex:
    """
    A convenient wrapper class around Faiss IndexIDMap2, supporting:
    - Building/loading a FlatL2 index with ID mapping.
    - Adding/removing/updating embeddings by ID.
    - Searching (full index or a subset of IDs).
    """

    def __init__(self, index_file: str, dimension: int, faiss_lock: Optional[Lock]):
        """
        :param index_file: Path to store/read the Faiss index.
        :param dimension: Embedding dimension (e.g., 768).
        :param faiss_lock: Multiprocessing lock for process-safe operations.
            Required for safe operation in a multiprocessing environment.
        """
        self.index_file = index_file
        self.dimension = dimension
        self.index: faiss.IndexIDMap2 = None  # The in-memory Faiss index
        self._data = {}  # In-memory dict of {id -> embedding} for subset building, etc.
        self.lock = faiss_lock
        
        if self.lock is None:
            logger.warning("Global Faiss lock is None. Running Faiss without locking. Be very certain you are in a single process environment.")

    # Don't call this method outside the class, you won't have proper locking
    def _build(self, embeddings: np.ndarray, ids: np.ndarray):
        """
        Build the index from scratch using the provided embeddings and IDs.
        Overwrites any existing in-memory data/index.
        """
        self._check_input_arrays(embeddings, ids)
        self._data.clear()

        # Store in memory
        for i, emb_id in enumerate(ids):
            self._data[emb_id] = embeddings[i]

        # Build Faiss index
        self.index = self._build_index_from_data()
        # Save the index to disk
        self.save()

    @with_faiss_lock(write_operation=False)
    def load_or_build_from_scratch(self):
        """
        Build a new index from scratch and add some random embeddings for initialization.
        """
        np.random.seed(42)
        num_embeddings = 0
        all_embeddings = np.random.rand(num_embeddings, self.dimension).astype(np.float32)
        all_ids = np.arange(num_embeddings, dtype=np.int64)
        self.load_or_build(all_embeddings, all_ids)

    def _load(self):
        """
        Load an existing index from disk.
        Returns True if successfully loaded, False otherwise.
        """
        try:
            logger.debug(f"Loading existing index from '{self.index_file}' ...")
            loaded_index = faiss.read_index(self.index_file)

            # Verify the loaded index is compatible
            if isinstance(loaded_index, faiss.IndexIDMap2):
                base_index = loaded_index.index
                if hasattr(base_index, 'd') and base_index.d == self.dimension:
                    self.index = loaded_index
                    logger.debug(f"Loaded index contains {self.index.ntotal} vectors.")
                    return True
                else:
                    logger.debug(f"Dimension mismatch in loaded index: got {getattr(base_index, 'd', 'unknown')}, expected {self.dimension}")
            else:
                logger.debug(f"Loaded index is not an IndexIDMap2: {type(loaded_index).__name__}")
            return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}.")
            return False

    @with_faiss_lock(write_operation=True)
    def load_or_build(self, embeddings: np.ndarray, ids: np.ndarray):
        """
        Load an existing index from disk (if available),
        otherwise build a new index from the provided data and save it.
        """
        if os.path.exists(self.index_file):
            if self._load():
                # Rebuild in-memory data
                self._data.clear()
                self._check_input_arrays(embeddings, ids)
                for i, emb_id in enumerate(ids):
                    self._data[emb_id] = embeddings[i]
                return
            else:
                logger.info("Rebuilding index from scratch...")
                self._build(embeddings, ids)
                return
        else:
            logger.info("No existing index found. Building from scratch...")
            self._check_input_arrays(embeddings, ids)
            self._data.clear()
            
            # Store in memory
            for i, emb_id in enumerate(ids):
                self._data[emb_id] = embeddings[i]
            
            # Build Faiss index
            self.index = self._build_index_from_data()
            self.save()

    def save(self):
        """ Save the current index to disk. """
        if self.index is not None:
            logger.debug(f"Saving {self.index.ntotal} indices to '{self.index_file}'...")
            faiss.write_index(self.index, self.index_file)
            logger.debug("Index saved.")
        else:
            logger.debug("No index to save.")
            
    def get_index_size(self):
        """ Get the size of the index. """
        if self.index is not None:
            return self.index.ntotal
        else:
            return 0

    @with_faiss_lock(write_operation=True)
    def add_embeddings(self, new_embeddings: np.ndarray, new_ids: np.ndarray):
        """
        Add new embeddings (and their IDs) to both the in-memory data and the Faiss index.
        If the index doesn't exist yet, this method builds it from scratch.
        """
        if new_embeddings.ndim == 1:
            # Single embedding case => reshape
            new_embeddings = new_embeddings[None, :]

        self._check_input_arrays(new_embeddings, new_ids)

        if self.index is None:
            # If there's no index, build from scratch
            self._check_input_arrays(new_embeddings, new_ids)
            self._data.clear()
            
            # Store in memory
            for i, emb_id in enumerate(new_ids):
                self._data[emb_id] = new_embeddings[i]
            
            # Build Faiss index
            self.index = self._build_index_from_data()
            self.save()
            return new_ids.tolist()

        # Update in-memory data
        for i, emb_id in enumerate(new_ids):
            self._data[emb_id] = new_embeddings[i]

        # Add to Faiss
        new_embeddings_f32 = new_embeddings.astype(np.float32)
        new_ids_int64 = new_ids.astype(np.int64)
        self.index.add_with_ids(new_embeddings_f32, new_ids_int64)
        self.save()
        return new_ids.tolist()

    @with_faiss_lock(write_operation=True)
    def remove_embeddings(self, ids_to_remove: np.ndarray):
        """
        Remove embeddings by ID from the index and in-memory data.
        """
        if self.index is None:
            logger.warning("Index not built yet. Nothing to remove.")
            return
        
        # Convert to int64
        ids_to_remove = ids_to_remove.astype(np.int64)

        removed_count = self.index.remove_ids(ids_to_remove)
        if removed_count < 0:
            logger.warning("Warning: removal may not be fully supported by this index type.")
        else:
            logger.debug(f"Requested removal of {len(ids_to_remove)} IDs. "
                  f"Index reported removed_count={removed_count}.")

        # Remove from in-memory data
        for emb_id in ids_to_remove:
            self._data.pop(emb_id, None)
        
        # save the index to disk
        self.save()

    @with_faiss_lock(write_operation=True)
    def update_embeddings(self, updated_embeddings: np.ndarray, updated_ids: np.ndarray):
        """
        Update existing embeddings by removing any old ones with the same IDs,
        then adding the new versions.
        If some of the IDs did not exist previously, they are simply added.
        """
        if updated_embeddings.ndim == 1:
            # Single embedding => reshape
            updated_embeddings = updated_embeddings[None, :]

        self._check_input_arrays(updated_embeddings, updated_ids)

        # If index doesn't exist yet, just add the embeddings
        if self.index is None:
            return self.add_embeddings(updated_embeddings, updated_ids)
            
        # Remove old
        ids_to_remove = updated_ids.astype(np.int64)
        removed_count = self.index.remove_ids(ids_to_remove)
        
        # Remove from in-memory data
        for emb_id in ids_to_remove:
            self._data.pop(emb_id, None)
            
        # Add new embeddings to memory
        for i, emb_id in enumerate(updated_ids):
            self._data[emb_id] = updated_embeddings[i]
            
        # Add to index
        updated_embeddings_f32 = updated_embeddings.astype(np.float32)
        updated_ids_int64 = updated_ids.astype(np.int64)
        self.index.add_with_ids(updated_embeddings_f32, updated_ids_int64)
        
        # Save to disk
        self.save()
        logger.debug(f"Updated {len(updated_ids)} embeddings.")

    @with_faiss_lock(write_operation=False)
    def search(self, query_vector: np.ndarray, k: int = 5, subset_ids: List[int] = None, exclude_ids: List[int] = None):
        """
        Search the index for the k nearest neighbors of a single query vector.
        Optionally restrict to a subset of IDs (include) and/or exclude specific IDs.
        
        :param query_vector: The query vector to search for
        :param k: Number of nearest neighbors to return
        :param subset_ids: Optional array of IDs to include in search (global scope)
        :param exclude_ids: Optional array of IDs to exclude from search (blacklist)
        """
        logger.debug(f"retrieve.search: query_vector: {query_vector.shape}, k: {k}, subset_ids: {subset_ids}")
        if k <= 0:
            logger.warning(f"retrieve.search: k is less than or equal to 0: {k}. Returning empty arrays.")
            # return empty 2d arrays
            return np.empty((1, 0)), np.empty((1, 0))
        
        if not self._load():
            raise ValueError("Failed to load index from disk.")
        
        query_vector = self._ensure_f32_2d(query_vector)

        # Use built-in FAISS selectors if either subset_ids or exclude_ids are provided
        if subset_ids is not None or exclude_ids is not None:
            sel = None
            
            if subset_ids is not None and exclude_ids is not None:
                # Both include and exclude: AND(include, NOT(exclude))
                include_sel = faiss.IDSelectorBatch(subset_ids)
                exclude_sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(exclude_ids))
                sel = faiss.IDSelectorAnd(include_sel, exclude_sel)
            elif subset_ids is not None:
                # Include only
                sel = faiss.IDSelectorBatch(subset_ids)
            elif exclude_ids is not None:
                # Exclude only
                sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(exclude_ids))
            
            params = faiss.SearchParametersIVF()
            params.sel = sel
            
            distances, found_ids = self.index.search(query_vector, k, params=params)
        else:
            # Search the full index
            if self.index is None:
                logger.warning("Index does not exist. Cannot perform search.")
                return None, None
            distances, found_ids = self.index.search(query_vector, k)

        return distances, found_ids

    # ---------------------- INTERNAL METHODS ----------------------

    def _build_index_from_data(self) -> faiss.IndexIDMap2:
        """
        Rebuilds a Faiss IndexIDMap2 from self._data.
        """
        all_ids = np.array(list(self._data.keys()), dtype=np.int64)
        all_embeddings = np.array(list(self._data.values()), dtype=np.float32)
        if all_embeddings.shape[0] == 0:
            # Build an empty index if no data
            flat_index = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIDMap2(flat_index)

        assert all_embeddings.shape[1] == self.dimension, (
            f"Dimension mismatch in _build_index_from_data(). "
            f"Expected {self.dimension}, got {all_embeddings.shape[1]}"
        )

        flat_index = faiss.IndexFlatL2(self.dimension)
        index_id_map = faiss.IndexIDMap2(flat_index)
        index_id_map.add_with_ids(all_embeddings, all_ids)
        logger.debug(f"Built a new index with {index_id_map.ntotal} vectors.")
        return index_id_map

    def _build_subset_index(self, subset_ids: np.ndarray) -> faiss.IndexIDMap2:
        """
        Build a temporary smaller index for just the given subset IDs
        by looking them up in self._data.
        """
        subset_ids = subset_ids.astype(np.int64)

        subset_embeddings = []
        filtered_ids = []
        for emb_id in subset_ids:
            emb = self._data.get(emb_id, None)
            if emb is not None:
                subset_embeddings.append(emb)
                filtered_ids.append(emb_id)

        if not subset_embeddings:
            # No matching IDs => empty index
            empty_index = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIDMap2(empty_index)

        subset_embeddings = np.array(subset_embeddings, dtype=np.float32)
        filtered_ids = np.array(filtered_ids, dtype=np.int64)

        flat_index = faiss.IndexFlatL2(self.dimension)
        subset_index = faiss.IndexIDMap2(flat_index)
        subset_index.add_with_ids(subset_embeddings, filtered_ids)
        return subset_index

    def _check_input_arrays(self, embeddings: np.ndarray, ids: np.ndarray):
        """
        Validate input embeddings and IDs for dimension consistency.
        Raises AssertionError if there's a mismatch.
        """
        assert embeddings.ndim == 2, (
            f"Embeddings must be 2D. Got shape {embeddings.shape}."
        )
        assert embeddings.shape[1] == self.dimension, (
            f"Dimension mismatch. Index dimension={self.dimension}, "
            f"embeddings dimension={embeddings.shape[1]}"
        )
        assert embeddings.shape[0] == ids.shape[0], (
            f"Embeddings row count ({embeddings.shape[0]}) does not match IDs length ({ids.shape[0]})."
        )

    @staticmethod
    def _ensure_f32_2d(arr: np.ndarray) -> np.ndarray:
        """Ensures input array is float32 and 2D (1, dim) for a single query vector."""
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr



if __name__ == "__main__":
    import time

    index_file = "my_faiss_full_index.index"
    dimension = 768
    num_embeddings = 100_000

    # Generate data
    logger.info("Generating random embeddings ...")
    np.random.seed(42)
    all_embeddings = np.random.rand(num_embeddings, dimension).astype(np.float32)
    all_ids = np.arange(num_embeddings, dtype=np.int64)

    # For test purposes, we don't need a lock in single process mode
    faiss_index = FaissIndex(index_file=index_file, dimension=dimension, faiss_lock=None)

    # Load or build
    faiss_index.load_or_build(all_embeddings, all_ids)

    # Subset search
    subset_size = 20_000
    subset_ids = np.random.choice(all_ids, subset_size, replace=False)
    query_vector = all_embeddings[0]  # single vector

    logger.info("\nPerforming subset search...")
    start = time.time()
    distances, found_ids = faiss_index.search(query_vector, k=5, subset_ids=subset_ids)
    logger.info(f"Subset search time: {time.time() - start:.6f} seconds")
    logger.info("Distances:", distances)
    logger.info("IDs:", found_ids)

    # Add a new embedding
    new_embedding = np.random.rand(1, dimension).astype(np.float32)
    new_id = np.array([999999], dtype=np.int64)
    faiss_index.add_embeddings(new_embedding, new_id)
    logger.info("\nAdded new embedding with ID=999999")

    # Update an existing embedding
    updated_embedding = np.random.rand(1, dimension).astype(np.float32)
    updated_id = np.array([0], dtype=np.int64)
    faiss_index.update_embeddings(updated_embedding, updated_id)
    logger.info("Updated embedding for ID=0")

    # Remove an embedding
    remove_id = np.array([10], dtype=np.int64)
    faiss_index.remove_embeddings(remove_id)
    logger.info("Removed embedding with ID=10")

    # Save changes to disk
    faiss_index.save()
