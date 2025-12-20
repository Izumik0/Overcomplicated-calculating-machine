import sys


def convert_line(line):
    # Only process ATOM or HETATM lines
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return line

    # Extract residue name (columns 17-20 in PDB standard)
    res_name = line[17:20].strip()

    # Only apply to DNA residues
    if res_name not in ["DA", "DT", "DC", "DG", "ADE", "THY", "CYT", "GUA"]:
        return line

    # Extract atom name (columns 12-16)
    atom_name = line[12:16].strip()

    # -- MAPPING LOGIC (New -> Old) --

    # 1. Sugar Hydrogens
    if atom_name == "H5'":
        new_name = "1H5'"
    elif atom_name == "H5''":
        new_name = "2H5'"
    elif atom_name == "H2'":
        new_name = "1H2'"
    elif atom_name == "H2''":
        new_name = "2H2'"

    # 2. Thymine (DT) Specifics
    elif res_name in ["DT", "THY"]:
        if atom_name == "C7":
            new_name = "C5M"
        elif atom_name == "H71":
            new_name = "H51"
        elif atom_name == "H72":
            new_name = "H52"
        elif atom_name == "H73":
            new_name = "H53"
        else:
            new_name = atom_name
    else:
        new_name = atom_name

    # If no change, return line
    if new_name == atom_name:
        return line

    # Reconstruct the PDB line with correct spacing (Atom name is cols 12-16)
    # <Left padding to 12 chars> <Atom Name padded to 4 chars> <Right side>
    # PDB format requires specific alignment for atom names:
    # 4-character names start at col 13 (index 12)
    # < 4-char names usually start at col 14 (index 13)

    before_atom = line[:12]
    after_atom = line[16:]

    # Formatting the new name to fit the 4-char field correctly
    if len(new_name) == 4:
        # e.g., "1H5'"
        formatted_name = new_name
    else:
        # e.g., "C5M" (needs a leading space to align correctly)
        formatted_name = " " + new_name.ljust(3)

    return before_atom + formatted_name + after_atom


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python new_to_old.py input.pdb output.pdb")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            f_out.write(convert_line(line))

    print(f"Converted {input_file} to {output_file} using Old Standard (CHARMM27) format.")