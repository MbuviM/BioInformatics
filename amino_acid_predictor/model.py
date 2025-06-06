"""
This project focuses on predicting an amino acid based on the codons given as input. 
It is a classification project that helps in understanding codons and amino acids better.
"""

# Import necessary libraries
import pandas as pd

# Create data dict
amino_acid_data = {
    "Histidine": ["CAU", "CAC"], 
    "Arginine": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"], 
    "Lysine": ["AAA", "AAG"],
    "Isoleucine": ["AUU", "AUC", "AUA"], 
    "Aspartic Acid": ["GAU", "GAC"],
    "Glumatic Acid":["GAA", "GAG"],
    "Threonine": ["ACU", "ACC", "ACA", "ACG"],
    "Methionine": ["AUG"], 
    "Glytamine": ["CAA", "CAG"],
    "Phenylanine": ["UUU", "UUC"],
    "Leucine": ["CUU", "CUC", "CUA", "CUG", "UUA", "UUG"],
    "Tryptophan": ["UGG"],
    "Alanine": ["GCU", "GCC", "GCA", "GCG"],
    "Tyrosine": ["UAU", "UAC"],
    "Cysteine": ["UGU", "UGC"],
    "Asparagine": ["AAU", "AAC"],
    "Valina": ["GUU", "GUC", "GUA", "GUG"],
    "Glycine": ["GGU", "GGC", "GGA", "GGG"],
    "Serine": ["AGU", "AGC", "UCU", "UCC", "UCA", "UCG"],
    "Proline": ["CCU", "CCC", "CCA", "CCG"]
}

# Create a new dictionary
data_rows = []
for amino_acids, codons in amino_acid_data.items():
    for codon in codons:
        data_rows.append({"codon": codon, "amino_acid": amino_acids})

#print(data_rows)

# Convert dict to pandas dataframe
def convert_dictionary(data):
    # index allows the values of the dictionary to be columns
    codons_amino_acid_dataset = pd.DataFrame.from_dict(data)
    return codons_amino_acid_dataset

dataset = data_rows
codons_amino_acid_dataset = convert_dictionary(dataset)

# Return the dataset
#print(codons_amino_acid_dataset)

# Print the first 5 rows
print(codons_amino_acid_dataset.head())