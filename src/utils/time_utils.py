

def get_sec_h_m_s(time_str):
    """Get seconds from time."""
    
    elements = time_str.split(':')
    if len(elements) == 3:
        h, m, s = elements
    else:
        h = 0
        m, s = elements
    return int(h) * 3600 + int(m) * 60 + int(s)