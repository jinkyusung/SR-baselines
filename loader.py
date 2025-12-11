import os
import torch
from pathlib import Path
from typing import List, Tuple, Dict


# ----------------------------- Temporal Dataset Loader ----------------------------- #


class TemporalDataset:
    def __init__(self, path: str, k_core: int = 10) -> None:
        """
        Initializes the dataset loader.

        Args:
            path (str): The file path to the raw interaction file (e.g., '.inter').
            k_core (int): The core number for filtering (e.g., 10). 
                          Set to 1 or 0 to disable filtering.
        """
        self.path = path
        self.k_core = k_core
        
        base_name = os.path.splitext(self.path)[0]
        self.cache_path = f"{base_name}_{self.k_core}.pt"

        self.data = torch.Tensor()
        self.start_time = float('inf')
        self.end_time = float('-inf')
        
        self.num_users = 0
        self.num_items = 0

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Loads raw data from the file specified in self.path.
        This method must be implemented by a child class.
        """
        raise NotImplementedError("Child class must implement _load_raw_data")

    def _load_from_cache(self) -> None:
        """Loads the processed dataset from a cached .pt file."""
        print(f"Loading from cache: {self.cache_path}")
        cached_data = torch.load(self.cache_path)
        self.data = cached_data['data']
        self.num_users = cached_data['num_users']
        self.num_items = cached_data['num_items']
        
        # [NOTE] The cache stores normalized data.
        # The original 'start_time' and 'end_time' (pre-normalization) 
        # are not re-loaded, as they are not needed for get_data().
        if self.data.shape[0] > 0:
            pass
        else:
            self.start_time = 0.0
            self.end_time = 0.0

    def _save_to_cache(self) -> None:
        """Saves the processed dataset to a .pt cache file."""
        print(f"Saving to cache: {self.cache_path}")
        data_to_save = {
            'data': self.data, # self.data is the normalized tensor
            'num_users': self.num_users,
            'num_items': self.num_items,
            # [NOTE] If needed, the original (pre-normalization) start/end times
            # 'start_time_original': self.start_time,
            # 'end_time_original': self.end_time
        }
        torch.save(data_to_save, self.cache_path)

    def _filter_k_core(self, data_ids: torch.Tensor) -> torch.Tensor:
        """
        Iteratively filters a dataset to meet the k-core constraint.
        [MODIFIED]
        - Now accepts an [N, 2] long ID tensor.
        - Returns an [N] boolean mask.

        A k-core dataset ensures that all remaining users and items 
        have at least k interactions. This method uses self.k_core.

        Args:
            data_ids (torch.Tensor): The interaction ID tensor (N, 2) 
                                     [uid, sid], must be torch.long.

        Returns:
            torch.Tensor: A boolean mask (shape N) where True indicates
                          an interaction to keep.
        """
        k = self.k_core
        if k <= 1:
            return torch.ones(data_ids.shape[0], dtype=torch.bool)

        print(f"Applying {k}-core filtering...")
        original_count = data_ids.shape[0]

        # Indices of the original data to keep track of
        current_indices = torch.arange(original_count)
        
        # Calculate max IDs based on original data (only once)
        num_users = int(data_ids[:, 0].max().item()) + 1
        num_items = int(data_ids[:, 1].max().item()) + 1
        
        while True:
            # Get only the IDs corresponding to the currently remaining indices
            current_data_ids = data_ids[current_indices]

            if current_data_ids.shape[0] == 0:
                print("Warning: k-core filtering removed all data.")
                break

            # bincount raises an error on empty tensors even with minlength
            user_degrees = torch.bincount(current_data_ids[:, 0], minlength=num_users)
            item_degrees = torch.bincount(current_data_ids[:, 1], minlength=num_items)

            user_mask = (user_degrees >= k)
            item_mask = (item_degrees >= k)

            # This mask is relative to current_data_ids (and thus current_indices)
            interaction_mask_relative = user_mask[current_data_ids[:, 0]] & item_mask[current_data_ids[:, 1]]

            if interaction_mask_relative.all():
                # No more filtering is needed
                break
            
            # Keep only the indices that survived this iteration
            current_indices = current_indices[interaction_mask_relative]
        
        # Create the final boolean mask
        final_mask = torch.zeros(original_count, dtype=torch.bool)
        final_mask[current_indices] = True # Mark True only for the surviving original indices
        
        filtered_count = final_mask.sum().item()
        print(f"Filtered {original_count - filtered_count} interactions. "
              f"Remaining: {filtered_count}")
        return final_mask

    def _preprocess(self) -> None:
        """
        Orchestrates the data preprocessing pipeline.
        [MODIFIED]
        - Fixes timestamp precision loss bug (loads as float32).
        - k-core filter logic is now mask-based (via _filter_k_core).
        - Fixes normalization order bug (calculates min/max *after* filtering).
        """
        if os.path.exists(self.cache_path):
            self._load_from_cache()
            return

        print(f"Cache not found. Processing raw data from: {self.path}")
        
        # 1. Load raw data (ignore pre-filter min/max)
        raw_data_list, _, _ = self._load_raw_data()

        if not raw_data_list:
            print("Warning: No data loaded.")
            return

        # 2. Convert to Float tensor (preserves timestamp precision)
        data_float_tensor = torch.tensor(raw_data_list, dtype=torch.float32)

        if data_float_tensor.shape[0] == 0:
            print("Warning: Data list was empty.")
            return

        # 3. --- K-Core Filtering Step ---
        # Extract IDs as long type to pass to the filter
        data_ids_long = data_float_tensor[:, :2].long()
        
        # Receive the boolean mask
        keep_mask = self._filter_k_core(data_ids_long)
        
        # Apply the mask to the original float tensor
        data_tensor = data_float_tensor[keep_mask]

        # 4. Handle cases where all data is filtered out
        if data_tensor.shape[0] == 0:
            print("Warning: All data was filtered out by k-core.")
            self.data = torch.empty((0, 3), dtype=torch.float64) # Save as float64
            self.start_time = 0.0
            self.end_time = 0.0
            self.num_users = 0
            self.num_items = 0
            self._save_to_cache() 
            return

        # 5. --- ID Remapping Step (on filtered data) ---
        uids_long = data_tensor[:, 0].long()
        unique_uids, remapped_uids = torch.unique(uids_long, return_inverse=True)
        data_tensor[:, 0] = remapped_uids.float() # Store remapped IDs back as float
        self.num_users = len(unique_uids)

        sids_long = data_tensor[:, 1].long()
        unique_sids, remapped_sids = torch.unique(sids_long, return_inverse=True)
        data_tensor[:, 1] = remapped_sids.float() # Store remapped IDs back as float
        self.num_items = len(unique_sids)

        # 6. --- Timestamp Normalization Step ---
        # Convert to float64 for final storage
        self.data = data_tensor.to(torch.float64) 
        
        # [BUG FIX] Calculate min/max based on the filtered and remapped data
        self.start_time = self.data[:, 2].min().item()
        self.end_time = self.data[:, 2].max().item()
        
        if self.end_time == self.start_time:
            self.data[:, 2] = 0.0
        else:
            # [BUG FIX] Normalize using the newly calculated start/time
            self.data[:, 2] = (self.data[:, 2] - self.start_time) / (self.end_time - self.start_time)
        
        # 7. Save to cache
        self._save_to_cache()
    
    def get_data(self) -> torch.Tensor:
        """
        Returns the processed data tensor (k-core filtered and remapped).
        """
        return self.data

    def get_shape(self) -> Tuple[int, int]:
        """
        Returns the total number of unique users and unique items
        after k-core filtering and remapping.
        """
        return (self.num_users, self.num_items)
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset statistics.
        Calculates user, item, and interaction counts, density, 
        and sequence length statistics (min, max, mean, variance).
        """
        if self.num_users == 0 or self.num_items == 0:
            return "Dataset is empty or not loaded."

        # 1. Basic Counts
        n_users = self.num_users
        n_items = self.num_items
        n_interactions = self.data.shape[0]
        dataset_name = os.path.basename(self.path)

        # 2. Sequence Length Statistics
        # We use bincount on user IDs to get the number of interactions per user.
        # Since IDs are remapped to [0, n_users-1], this is efficient.
        user_ids = self.data[:, 0].long()
        user_degrees = torch.bincount(user_ids, minlength=n_users).float()

        min_seq = user_degrees.min().item()
        max_seq = user_degrees.max().item()
        mean_seq = user_degrees.mean().item()
        var_seq = user_degrees.var().item()  # Unbiased sample variance

        # 3. Density Calculation
        # Density = Interactions / (Users * Items)
        matrix_size = n_users * n_items
        density = n_interactions / matrix_size if matrix_size > 0 else 0.0

        # 4. Formatted Output
        header = "=" * 50
        divider = "-" * 50
        
        stats_str = (
            f"\n{header}\n"
            f"Dataset Statistics: {dataset_name}\n"
            f"{divider}\n"
            f"{'Number of Users':<25} : {n_users}\n"
            f"{'Number of Items':<25} : {n_items}\n"
            f"{'Number of Interactions':<25} : {n_interactions}\n"
            f"{'Density':<25} : {density:.5f} ({density*100:.3f}%)\n"
            f"{divider}\n"
            f"Sequence Length Statistics:\n"
            f"  - Min Length            : {min_seq:.0f}\n"
            f"  - Mean Length           : {mean_seq:.2f}\n"
            f"  - Max Length            : {max_seq:.0f}\n"
            f"  - Variance              : {var_seq:.2f}\n"
            f"{header}\n"
        )
        
        return stats_str


# ----------------------------- Gowalla Dataset Loader ----------------------------- #


class Gowalla(TemporalDataset):
    def __init__(self, path: str = 'dataset/gowalla/gowalla.inter', k_core: int = 5):
        """
        Initializes the Gowalla dataset loader.

        Args:
            path (str): The file path to the 'gowalla.inter' file.
            k_core (int): The k-core value for filtering.
        """
        super().__init__(path, k_core)
        self._preprocess()

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Implements the data loading logic for the Gowalla .inter file format.
        
        It reads the file, skips the header, and extracts the
        user ID, item ID, and timestamp from each line.

        Returns:
            Tuple[List[List[float]], float, float]:
                - A list of raw interactions as [uid, sid, timestamp].
                - The earliest timestamp.
                - The latest timestamp.
        """
        data = []
        start_time = float('inf')
        end_time = float('-inf')

        with open(self.path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                
                parts = list(map(float, line.strip().split()))
                uid, sid, timestamp = parts[0], parts[1], parts[2]
                
                data.append([uid, sid, timestamp])

                if timestamp <= start_time:
                    start_time = timestamp
                if timestamp >= end_time:
                    end_time = timestamp

        return data, start_time, end_time


# ----------------------------- Steam Dataset Loader ----------------------------- #


class Steam(TemporalDataset):
    def __init__(self, path: str = 'dataset/steam/steam.inter', k_core: int = 5):
        """
        Initializes the Steam dataset loader.

        Args:
            path (str): The file path to the 'steam.inter' file.
        """
        super().__init__(path, k_core)
        self._preprocess()

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Implements the data loading logic for the Steam .inter file format.
        
        It reads the tab-separated file, skips the header, and extracts 
        the user ID (col 0), product ID (col 3), and timestamp (col 5).

        Returns:
            Tuple[List[List[float]], float, float]:
                - A list of raw interactions as [uid, sid, timestamp].
                - The earliest timestamp.
                - The latest timestamp.
        """
        data = []
        start_time = float('inf')
        end_time = float('-inf')

        with open(self.path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                
                # Assumes standard .inter format (Tab-Separated)
                parts = line.strip().split('\t')
                
                # Schema mapping:
                # user_id:token    -> parts[0]
                # product_id:token -> parts[3]
                # timestamp:float  -> parts[5]
                
                uid = float(parts[0])
                sid = float(parts[3])
                timestamp = float(parts[5])
                
                data.append([uid, sid, timestamp])

                if timestamp <= start_time:
                    start_time = timestamp
                if timestamp >= end_time:
                    end_time = timestamp

        return data, start_time, end_time


# ----------------------------- MovieLens Dataset Loader ----------------------------- #


class MovieLens(TemporalDataset):
    def __init__(self, path: str, k_core: int = 5):
        """
        Initializes the MovieLens dataset loader.

        Args:
            path (str): The file path to the 'ml-XX.inter' file.
        """
        super().__init__(path, k_core)
        self._preprocess()

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Implements the data loading logic for the Steam .inter file format.
        
        It reads the tab-separated file, skips the header, and extracts 
        the user ID (col 0), product ID (col 3), and timestamp (col 5).

        Returns:
            Tuple[List[List[float]], float, float]:
                - A list of raw interactions as [uid, sid, timestamp].
                - The earliest timestamp.
                - The latest timestamp.
        """
        data = []
        start_time = float('inf')
        end_time = float('-inf')

        with open(self.path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                
                # Assumes standard .inter format (Tab-Separated)
                parts = line.strip().split('\t')
                
                uid = float(parts[0])
                sid = float(parts[1])
                timestamp = float(parts[3])
                
                data.append([uid, sid, timestamp])

                if timestamp <= start_time:
                    start_time = timestamp
                if timestamp >= end_time:
                    end_time = timestamp

        return data, start_time, end_time
    

# ----------------------------- Tmall Dataset Loader ----------------------------- #


class Tmall(TemporalDataset):
    def __init__(self, path: str, k_core: int = 5):
        """
        Initializes the Tmall dataset loader.

        Args:
            path (str): The file path to the 'tmall-<subtype>.inter' file.
        """
        super().__init__(path, k_core)
        self._preprocess()

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Implements the data loading logic for the Steam .inter file format.
        
        It reads the tab-separated file, skips the header, and extracts 
        the user ID (col 0), product ID (col 3), and timestamp (col 5).

        Returns:
            Tuple[List[List[float]], float, float]:
                - A list of raw interactions as [uid, sid, timestamp].
                - The earliest timestamp.
                - The latest timestamp.
        """
        data = []
        start_time = float('inf')
        end_time = float('-inf')

        with open(self.path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                
                # Assumes standard .inter format (Tab-Separated)
                parts = line.strip().split('\t')
                
                uid = float(parts[0])
                sid = float(parts[2])
                timestamp = float(parts[4])
                
                data.append([uid, sid, timestamp])

                if timestamp <= start_time:
                    start_time = timestamp
                if timestamp >= end_time:
                    end_time = timestamp

        return data, start_time, end_time


# ----------------------------- Amazon Dataset Loader ----------------------------- #


class Amazon(TemporalDataset):
    def __init__(self, path: str, k_core: int = 5):
        """
        Initializes the Amazon dataset loader.

        Args:
            path (str): The file path to the Amazon .inter file.
            k_core (int): The core number for filtering.
        """
        super().__init__(path, k_core=k_core)
        self._preprocess()

    def _load_raw_data(self) -> Tuple[List[List[float]], float, float]:
        """
        Implements the data loading logic for the Amazon .inter file format.
        
        It reads the tab-separated file, skips the header, and maps
        the alphanumeric user/item IDs to intermediate integer IDs.
        It extracts user ID (col 0), item ID (col 1), and timestamp (col 3).

        Returns:
            Tuple[List[List[float]], float, float]:
                - A list of raw interactions as [uid_int, sid_int, timestamp].
                - The earliest timestamp.
                - The latest timestamp.
        """
        data = []
        start_time = float('inf')
        end_time = float('-inf')

        # Dictionaries for mapping string IDs to temporary integer IDs
        user_str_to_int: Dict[str, int] = {}
        item_str_to_int: Dict[str, int] = {}
        next_user_int_id = 0
        next_item_int_id = 0

        with open(self.path, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                
                parts = line.strip().split('\t')
                
                # Schema mapping:
                # user_id:token (str)  -> parts[0]
                # item_id:token (str)  -> parts[1]
                # timestamp:float      -> parts[3]
                
                user_str = parts[0]
                item_str = parts[1]
                timestamp = float(parts[3])

                # --- String-to-Integer ID Mapping ---
                
                # Get or create integer ID for user
                if user_str not in user_str_to_int:
                    user_str_to_int[user_str] = next_user_int_id
                    next_user_int_id += 1
                uid_int = user_str_to_int[user_str]
                
                # Get or create integer ID for item
                if item_str not in item_str_to_int:
                    item_str_to_int[item_str] = next_item_int_id
                    next_item_int_id += 1
                sid_int = item_str_to_int[item_str]
                
                # -------------------------------------

                data.append([uid_int, sid_int, timestamp])

                if timestamp <= start_time:
                    start_time = timestamp
                if timestamp >= end_time:
                    end_time = timestamp

        return data, start_time, end_time


# ----------------------------- Temporal Adjacency Matrix ----------------------------- #


class TemporalAdjacencyMatrix:
    def __init__(self, data: torch.Tensor):
        """
        Args: 
            data (torch.Tensor): [N, 3] shape tensor containing
                                 [uid, sid, timestamp] for N interactions.
                                 IDs (uid, sid) must be 0-indexed and contiguous.
                                 timestamp values must be normalized between 0.0 and 1.0.
        """
        if data.shape[1] != 3:
            raise ValueError(f"Input data must have 3 columns [uid, sid, timestamp], but got {data.shape[1]}")

        self.indices = data[:, :2].t().long()
        self.timestamps = data[:, 2]
        
        num_interactions = data.shape[0]
        self.values = torch.ones(num_interactions, dtype=torch.float)

        num_users = int(data[:, 0].max().item()) + 1
        num_items = int(data[:, 1].max().item()) + 1
        self.shape = (num_users, num_items)

    def __call__(self, t: float) -> torch.Tensor:
        """
        Filters interactions up to time t and returns a sparse matrix.

        Args:
            t (float): Time threshold (between 0.0 and 1.0).
            
        Returns:
            torch.Tensor: Sparse COO tensor of shape (num_users, num_items)
                          where matrix[u, i] = 1 if an interaction exists <= t,
                          and 0 otherwise.
        """
        mask = self.timestamps <= t
        filtered_indices = self.indices[:, mask]
        filtered_values = self.values[mask]

        adj_matrix = torch.sparse_coo_tensor(
            indices=filtered_indices,
            values=filtered_values,
            size=self.shape,
            dtype=torch.float
        )
        return adj_matrix.coalesce()
    

# ----------------------------- class matcher ----------------------------- #


def load_data_object(dataset: str, k_core: int):

    amazon_dataset = {
        'amazon-books',
        'amazon-beauty',
        'amazon-toys',
        'amazon-electronics'
    }

    movielens_dataset = {
        'ml-1m', 'ml-100k'
    }

    steam_dataset = {
        'steam'
    }

    gowalla_dataset = {
        'gowalla'
    }

    tmall_dataset = {
        'tmall-buy',
        'tmall-click'
    }

    raw_file = Path(__file__).resolve().parent / 'dataset' / f'{dataset}.inter'

    if dataset in amazon_dataset:
        return Amazon(path=raw_file, k_core=k_core)

    elif dataset in steam_dataset:
        return Steam(path=raw_file, k_core=k_core)

    elif dataset in movielens_dataset:
        return MovieLens(path=raw_file, k_core=k_core)

    elif dataset in gowalla_dataset:
        return Gowalla(path=raw_file, k_core=k_core)

    elif dataset in tmall_dataset:
        return Tmall(path=raw_file, k_core=k_core)

    else:
        NotImplementedError(f"Dataset class for '{dataset}' is not explicited.")
