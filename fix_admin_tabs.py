"""
Script to wrap admin-only tabs (Reco_Masiva, Valida_Retro, ComparaRetroEntreSistema) 
with if is_admin: blocks and properly indent their content.
"""

def fix_admin_tabs_indentation():
    file_path = "src/ui/dashboard.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find line numbers for the three sections
    target_lines = {
        "TAB 9": None,      # with Reco_Masiva:
        "TAB 10": None,     # with Valida_Retro:
        "TAB 11": None,     # with ComparaRetroEntreSistema:
    }
    
    # Find the exact line numbers
    for i, line in enumerate(lines):
        if "# TAB 9: RECOMENDACIÓN MASIVA" in line:
            # Next non-empty line should be "with Reco_Masiva:"
            for j in range(i, min(i + 5, len(lines))):
                if "with Reco_Masiva:" in lines[j]:
                    target_lines["TAB 9"] = j
                    break
        elif "# TAB 10: VALIDACIÓN RETROSPECTIVA" in line:
            for j in range(i, min(i + 5, len(lines))):
                if "with Valida_Retro:" in lines[j]:
                    target_lines["TAB 10"] = j
                    break
        elif "# TAB 11: COMPARATIVA RETROSPECTIVA" in line:
            for j in range(i, min(i + 5, len(lines))):
                if "with ComparaRetroEntreSistema:" in lines[j]:
                    target_lines["TAB 11"] = j
                    break
    
    print(f"TAB 9 with line: {target_lines['TAB 9']}")
    print(f"TAB 10 with line: {target_lines['TAB 10']}")
    print(f"TAB 11 with line: {target_lines['TAB 11']}")
    
    # Find where each section ends (start of next "# ==========")
    section_ends = {}
    for tab_num, with_line in target_lines.items():
        if with_line is not None:
            # Search forward for the next comment section
            for j in range(with_line + 1, len(lines)):
                if "# ==========" in lines[j]:
                    section_ends[tab_num] = j - 1
                    break
            # If not found, it's the last section
            if tab_num not in section_ends:
                section_ends[tab_num] = len(lines) - 1
    
    print(f"Section ends: {section_ends}")
    
    # Process in reverse order to avoid line number shifts
    for tab_num in ["TAB 11", "TAB 10", "TAB 9"]:
        with_line = target_lines[tab_num]
        end_line = section_ends[tab_num]
        
        if with_line is None:
            continue
        
        print(f"\nProcessing {tab_num}: lines {with_line} to {end_line}")
        
        # Get the indent level of the "with" line
        with_indent = len(lines[with_line]) - len(lines[with_line].lstrip())
        content_indent = with_indent + 4  # Increase indent by 4 spaces
        
        # 1. Insert "if is_admin:" before the "with" line
        if_statement = " " * with_indent + "if is_admin:\n"
        lines.insert(with_line, if_statement)
        
        # 2. Increase indent of all lines from (with_line + 1) to (end_line + 1)
        # because we inserted a line
        with_line_updated = with_line + 1
        end_line_updated = end_line + 1
        
        for j in range(with_line_updated, end_line_updated + 1):
            if j < len(lines) and lines[j].strip():  # Only indent non-empty lines
                lines[j] = " " * 4 + lines[j]
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\n✅ File updated successfully!")

if __name__ == "__main__":
    fix_admin_tabs_indentation()
