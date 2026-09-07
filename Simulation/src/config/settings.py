import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Global Parameters
NUM_HOUSEHOLDS = 100
NUM_FIRMS = 50
NUM_GOODS = 10

# -------------------------------------------------------------------------
# 1. Leontief Input-Output Matrix (A)
# We generate a 10x10 non-negative matrix and ensure its spectral radius < 1
# -------------------------------------------------------------------------
_A_raw = np.random.rand(NUM_GOODS, NUM_GOODS)
# Calculate eigenvalues
_eigenvalues_A = np.linalg.eigvals(_A_raw)
_spectral_radius_A = np.max(np.abs(_eigenvalues_A))

# Scale to ensure spectral radius is less than 1 (e.g., 0.9)
LEONTIEF_MATRIX = (_A_raw / _spectral_radius_A) * 0.9

# Verify it
assert np.max(np.abs(np.linalg.eigvals(LEONTIEF_MATRIX))) < 1.0, "Leontief matrix spectral radius >= 1"


# -------------------------------------------------------------------------
# 2. Capital Accounts Matrix (K_req)
# Represents the amount of each capital good required to produce one unit 
# of another good. Dimensions: (NUM_GOODS, NUM_GOODS)
# -------------------------------------------------------------------------
CAPITAL_ACCOUNTS_MATRIX = np.random.rand(NUM_GOODS, NUM_GOODS) * 0.5


# -------------------------------------------------------------------------
# 3. Labor Use Vector (L_req)
# Amount of labor required to produce one unit of each good.
# Dimensions: (NUM_GOODS, 1)
# -------------------------------------------------------------------------
LABOR_USE_VECTOR = np.random.rand(NUM_GOODS, 1) * 0.2 + 0.1 # bounded strictly positive


# -------------------------------------------------------------------------
# 4. Endowments
# -------------------------------------------------------------------------
# Total labor endowment in the economy
TOTAL_LABOR_ENDOWMENT = 1000.0

# Total capital endowment for each good. Dimensions: (NUM_GOODS, 1)
TOTAL_CAPITAL_ENDOWMENT = np.ones((NUM_GOODS, 1)) * 500.0

