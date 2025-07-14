import re

def update_kenteken_format(kenteken_value):
    """
    Format a Dutch license plate (kenteken) string.
    
    Args:
        kenteken_value (str): The input kenteken string to format
        
    Returns:
        str: The formatted kenteken string
    """
    # Skip formatting for Android (this was a browser-specific check in JS)
    # In Python context, we'll just process all inputs
    
    original_value = kenteken_value
    tmp = kenteken_value
    
    # Remove spaces and hyphens
    tmp = tmp.replace(" ", "").replace("-", "")
    
    i = 0
    total_kenteken = ""
    
    # First pass: add hyphens between numeric/non-numeric transitions
    while i < len(tmp):
        # Count existing "is" patterns (this seems to be counting hyphens in the original)
        count1 = len(re.findall(r'is', total_kenteken))
        
        if (i + 1 < len(tmp) and 
            tmp[i].isdigit() != tmp[i + 1].isdigit() and 
            tmp[i] != "-" and tmp[i + 1] != "-" and 
            count1 < 2):
            total_kenteken += tmp[i] + "-"
        else:
            total_kenteken += tmp[i]
        i += 1
    
    # Second pass: handle 4-character segments
    count2 = len(re.findall(r'is', total_kenteken))
    if count2 < 2:
        tmp_parts = total_kenteken.split("-")
        for i2 in range(len(tmp_parts)):
            if len(tmp_parts[i2]) == 4:
                # Split 4-character segment: XX-XX
                old_segment = tmp_parts[i2]
                new_segment = old_segment[0] + old_segment[1] + "-" + old_segment[2] + old_segment[3]
                total_kenteken = total_kenteken.replace(old_segment, new_segment)
    
    # Remove leading hyphen if present
    if total_kenteken.startswith("-"):
        total_kenteken = total_kenteken[1:]
    
    # Clean up multiple segments (limit to 3 parts: XX-XX-XX)
    tmp_parts = total_kenteken.split("-")
    if len(tmp_parts) >= 4:
        total_kenteken = tmp_parts[0] + "-" + tmp_parts[1] + "-"
        for i in range(2, len(tmp_parts)):
            total_kenteken += tmp_parts[i]
    
    return total_kenteken.upper()

def is_numeric(char):
    """Helper function to check if a character is numeric (equivalent to $.isNumeric)"""
    return char.isdigit()