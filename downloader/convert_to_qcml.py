import h5py
import numpy as np
import pandas as pd
import qcelemental as qcel
from qcelemental.models import Molecule
from tqdm import tqdm


def process_hdf5_to_monomer_extended(
    hdf5_path="SPICE.hdf5", output_path="monomer_extended.pkl"
):
    """
    Process SPICE HDF5 file into monomer_extended.pkl format.

    This function reads the SPICE HDF5 file and converts each conformation into a row
    in a pandas DataFrame with the following columns:
    - qcel_molecule: QCElemental Molecule object
    - volume ratios: List of atomic volume ratios
    - valence widths: List of atomic valence widths
    - radial moments <r^2>: List of <r^2> moments
    - radial moments <r^3>: List of <r^3> moments
    - radial moments <r^4>: List of <r^4> moments
    - Z: Atomic numbers array
    - R: Cartesian coordinates in Angstroms (converted from Bohr)
    - cartesian_multipoles: Cartesian multipoles from MBIS analysis
    - TQ: Total charge
    - molecular_multiplicity: Molecular multiplicity

    Parameters:
    -----------
    hdf5_path : str
        Path to the SPICE.hdf5 file
    output_path : str
        Path to save the output pickle file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing processed data
    """

    rows = []
    charge_mismatches = []

    with h5py.File(hdf5_path, "r") as f:
        # Iterate through all molecules in the HDF5 file
        for mol_id in tqdm(f.keys(), desc="Processing molecules"):
            entry = f[mol_id]

            # Extract data that's constant across all conformations
            atomic_numbers = entry["atomic_numbers"][...]
            n_atoms = len(atomic_numbers)

            # Extract subset information
            subset = (
                entry["subset"][...][0].decode("utf-8")
                if "subset" in entry
                else "Unknown"
            )

            # Get number of conformations
            conformations = entry["conformations"][...]  # shape: (n_conf, n_atoms, 3)
            n_conformations = conformations.shape[0]

            # Extract MBIS data
            mbis_charges = entry["mbis_charges"][...]  # shape: (n_conf, n_atoms, 1)
            mbis_dipoles = entry["mbis_dipoles"][...]  # shape: (n_conf, n_atoms, 3)
            mbis_quadrupoles = entry["mbis_quadrupoles"][
                ...
            ]  # shape: (n_conf, n_atoms, 3, 3)

            # Process each conformation
            for conf_idx in range(n_conformations):
                # Convert coordinates from Bohr to Angstrom
                coords_bohr = conformations[conf_idx]  # shape: (n_atoms, 3)
                coords_angstrom = coords_bohr * qcel.constants.bohr2angstroms

                # Build cartesian multipoles array
                # Format: [charge, dipole_x, dipole_y, dipole_z,
                #          quadrupole_xx, quadrupole_xy, quadrupole_xz,
                #          quadrupole_yy, quadrupole_yz, quadrupole_zz]
                cartesian_multipoles = np.zeros((n_atoms, 10))

                for atom_idx in range(n_atoms):
                    # Charge (monopole)
                    cartesian_multipoles[atom_idx, 0] = mbis_charges[
                        conf_idx, atom_idx, 0
                    ]

                    # Dipoles
                    cartesian_multipoles[atom_idx, 1:4] = mbis_dipoles[
                        conf_idx, atom_idx, :
                    ]

                    # Quadrupoles (symmetric tensor, 6 unique components)
                    # Store as: xx, xy, xz, yy, yz, zz
                    Q = mbis_quadrupoles[conf_idx, atom_idx, :, :]
                    cartesian_multipoles[atom_idx, 4] = Q[0, 0]  # xx
                    cartesian_multipoles[atom_idx, 5] = Q[0, 1]  # xy
                    cartesian_multipoles[atom_idx, 6] = Q[0, 2]  # xz
                    cartesian_multipoles[atom_idx, 7] = Q[1, 1]  # yy
                    cartesian_multipoles[atom_idx, 8] = Q[1, 2]  # yz
                    cartesian_multipoles[atom_idx, 9] = Q[2, 2]  # zz

                # Create QCElemental Molecule object
                # Note: geometry expects 2D array (n_atoms, 3) in Angstroms
                # Let qcelemental auto-detect charge and multiplicity
                qcel_mol = Molecule(
                    symbols=[int(z) for z in atomic_numbers], geometry=coords_angstrom
                )

                # Placeholder values for fields not available in SPICE HDF5
                # These would need to be computed separately if needed
                volume_ratios = [0.0] * n_atoms  # Placeholder
                valence_widths = [0.0] * n_atoms  # Placeholder
                radial_moments_r2 = [0.0] * n_atoms  # Placeholder
                radial_moments_r3 = [0.0] * n_atoms  # Placeholder
                radial_moments_r4 = [0.0] * n_atoms  # Placeholder

                # Calculate sum of MBIS charges - this is the true molecular charge
                mbis_charge_sum = np.sum(cartesian_multipoles[:, 0])
                # Round to nearest integer for molecular charge
                molecular_charge = int(round(mbis_charge_sum))
                qcel_charge = int(qcel_mol.molecular_charge)

                # Check if MBIS sum matches the rounded value (within 0.1e tolerance)
                charge_diff = abs(mbis_charge_sum - molecular_charge)
                if charge_diff > 0.1:
                    charge_mismatches.append(
                        {
                            "mol_id": mol_id,
                            "conf_idx": conf_idx,
                            "molecular_charge": molecular_charge,
                            "mbis_charge_sum": mbis_charge_sum,
                            "qcel_charge": qcel_charge,
                            "difference": charge_diff,
                        }
                    )

                # Create row dictionary
                row = {
                    "qcel_molecule": qcel_mol,
                    "volume ratios": volume_ratios,
                    "valence widths": valence_widths,
                    "radial moments <r^2>": radial_moments_r2,
                    "radial moments <r^3>": radial_moments_r3,
                    "radial moments <r^4>": radial_moments_r4,
                    "Z": atomic_numbers.astype(np.int64),
                    "R": coords_angstrom.astype(np.float64),
                    "cartesian_multipoles": cartesian_multipoles.astype(np.float64),
                    "TQ": molecular_charge,  # Total charge from MBIS (rounded)
                    "molecular_multiplicity": int(
                        qcel_mol.molecular_multiplicity
                    ),  # From auto-detection
                    "subset": subset,  # Dataset subset information
                }

                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to pickle
    df.to_pickle(output_path)
    print(f"\nSaved {len(df)} conformations to {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Print charge verification statistics
    print("\n" + "=" * 60)
    print("CHARGE VERIFICATION REPORT")
    print("=" * 60)
    print("TQ values are set to round(sum(MBIS charges))")
    if charge_mismatches:
        print(
            f"WARNING: Found {len(charge_mismatches)} conformations with charge mismatches (>0.1e difference)"
        )
        print("These indicate MBIS charges don't sum to an integer (unusual)")
        print("\nFirst 10 mismatches:")
        for i, mismatch in enumerate(charge_mismatches[:10]):
            print(
                f"  {i + 1}. Mol {mismatch['mol_id']}, Conf {mismatch['conf_idx']}: "
                f"TQ={mismatch['molecular_charge']}, MBIS={mismatch['mbis_charge_sum']:.4f}, "
                f"QCel={mismatch['qcel_charge']}, Diff={mismatch['difference']:.4f}"
            )
    else:
        print("SUCCESS: All MBIS charges sum to integers within 0.1e tolerance!")

    # Print TQ distribution
    print("\nMolecular charge (TQ) distribution:")
    tq_counts = df["TQ"].value_counts().sort_index()
    for tq, count in tq_counts.items():
        print(f"  TQ={tq:+2d}: {count} conformations")

    # Print subset statistics
    print("\n" + "=" * 60)
    print("SUBSET DISTRIBUTION")
    print("=" * 60)
    subset_counts = df["subset"].value_counts()
    for subset_name, count in subset_counts.items():
        print(f"  {subset_name}: {count} conformations")

    return df


if __name__ == "__main__":
    # Test the function
    df = process_hdf5_to_monomer_extended()

    # Show summary
    print("\n" + "=" * 60)
    print("Summary of first entry:")
    print("=" * 60)
    first = df.iloc[0]
    for col in df.columns:
        val = first[col]
        print(f"\n{col}:")
        print(f"  Type: {type(val)}")
        if hasattr(val, "shape"):
            print(f"  Shape: {val.shape}")
            if val.size <= 20:
                print(f"  Data: {val}")
        else:
            print(f"  Value: {val}")
