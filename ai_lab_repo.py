import warnings
from ai_lab.__main__ import main

warnings.warn(
    "ai_lab_repo.py has moved to the ai_lab package. Use 'python -m ai_lab' instead. "
    "This compatibility shim will be removed after July 2025.",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
