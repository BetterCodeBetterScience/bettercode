"""Project constants.

This module defines physical and mathematical constants used throughout the project.
"""

# speed of light in a vacuum (m/s)
C = 299792458


class Constants:
    """Container class for physical constants with immutable attributes.
    
    Attributes
    ----------
    C : int
        Speed of light in a vacuum (m/s)
    """
    
    C = 299792458
    
    def __setattr__(self, name: str, value: object) -> None:
        """Prevent modification of constant values.
        
        Parameters
        ----------
        name : str
            Attribute name
        value : object
            Attempted new value
            
        Raises
        ------
        AttributeError
            Always raised to prevent modification
        """
        raise AttributeError("Constants cannot be modified")
