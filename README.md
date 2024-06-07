# shoulder

# Installation

## User

Simply use conda to create an environment from the `environment.yml` by running the following command:

```bash
conda create -f environment.yml
```

## Developer

If you plan to modify the `biorbd` [https://github.com/pyomeca/biorbd](https://github.com/pyomeca/biorbd) core, you will need to install the dependencies without installing `biorbd` from conda. 
The following command does most of it:

```bash
conda install pkgconfig cmake swig numpy scipy matplotlib rbdl eigen ipopt pyqt pyomeca vtk timyxml -cconda-forg
```

Please note that `bioviz` does not need to be installed. 
If you initialize the submodule, then the `PYTHONPATH` should points to `{$ROOT_SHOULDER}/external/bioviz`. 

## Scapula averaging

In order to run the `scapula_avering.py` script, you must extract the scapulas. The expected hierachy is `{$ROOT_SHOULDER}/run/scapula/{EXTRACTED_FOLDER}`, where `EXTRACTED_FOLDER` is either `Modele_stat` or `Scapula-BD-EOS` the subfolders are the extracted ones.

# Algorithms

In this section, I walk through the algorithms used in the `shoulder` package.

## Kinematic analysis
TODO

## Muscle analysis
TODO

## Optimize resting
TODO

## Scapula averaging 

### Some definitions

- <a id="definitions-aa"></a>**AA**: Acromion Angle
- <a id="definitions-ac"></a>**AC**: Dorsal of Acromioclavicular joint
- <a id="definitions-ai"></a>**AI**: Angulus Inferior joint
- <a id="definitions-gc_contours"></a>**GC_CONTOURS**: Glenoid Cavity Contours
- <a id="definitions-gc_mid"></a>**GC_MID**: Glenoid cavity midpoint computed from [*IE*](#definitions-ie) and [*SE*](#definitions-se)
- <a id="definitions-gc_normal"></a>**GC_NORMAL**: Glenoid cavity normal computed from the best fitting plane of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_circle_center"></a>**GC_CIRCLE_CENTER**: Glenoid cavity as the center point of the best fitting circle of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_center"></a>**GC_ELLIPSE_CENTER**: Glenoid cavity as the center point of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_major"></a>**GC_ELLIPSE_MAJOR**: Glenoid cavity as the major axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_minor"></a>**GC_ELLIPSE_MINOR**: Glenoid cavity as the minor axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-ie"></a>**IE**: Inferior Edge of glenoid
- <a id="definitions-se"></a>**SE**: Superior Edge of glenoid
- <a id="definitions-ts"></a>**TS**: Trigonum Spinae

- <a id="definitions-raw_data"></a>**Raw data**: The raw data is value directly extracted from the data file (.ply, .stl, .obj, etc.). 
- <a id="definitions-normalized_data"></a>**Normalized data**: The normalized data is the raw data that has been normalized so that the scapula is shown in a box of roughly 1x1x1 centered and rotated to be aligned with the ["ISB"](#definitions-isb) reference system.
 
- <a id="definitions-isb"></a>**ISB**: International Society of Biomechanics
- <a id="definitions-jcs"></a>**JCS**: Joint coordinate system, that is a 4x4 homogeneous matrix that describes the position and orientation of a joint in space.

### The algorithm

#### ***Step 1***: Load the reference scapula geometry

    In brief, load the geometry and locate the bony landmarks on the scapula.

    Main function called: 
        Scapula.from_landmarks(file_path, [prepointed_landmarks])
    where 
        file_path, The path to the scapula geometry file;
        prepointed_landmarks, Optional dictionary of the prepointed landmarks.
    
    Output of the Step 1:
        The scapula geometry
        Position of bony landmarks on that geometry

1. *Load data:* First, we load the reference scapula geometry from an object file (`*.ply`). For the statistical scapula, the reference scapula is the average asymptomatic scapula (A). For the EOS scapula, the reference scapula is a random scapula from the database (i.e., the file noted 001). These data are thereafter referred to as the [raw data](#definitions-raw_data).

2. *Roughly normalize the data:* From the [raw data](#definitions-raw_data), perform a rough normalization so the scapula is contained within a unit box. To do so, the min and max values of all the coordinates are extracted and the data are divided by the norm of the min-to-max vector.
*Note: this normalization is temporary and will be refined later on using the actual landmarks (see [*Step 3*](#method-step_3_normalize)).*

3. *Get bony landmarks:* Showing these roughly normalized data on screen, the user is requested to point the bony landmarks of the scapula, that is [*AA*](#definitions-aa), [*AC*](#definitions-ac), [*AI*](#definitions-ai), [*IE*](#definitions-ie), [*SE*](#definitions-se), [*GC_CONTOURS*](#definitions-gc_contours), [*TS*](#definitions-ts). For the [*GC_CONTOURS*](#definitions-gc_contours), the user is also asked to inform how many points should be pointed on the glenoid cavity before starting to actually point them on the scapula.
*Note: In order to reduce the developing time, these results were manually saved in the main script (i.e., the **prepointed_landmarks** when calling the `Scapula.from_landmarks` method) to prevent having to perform this pointing step each time the main script is run. In that case, for the [*GC_CONTOURS*](#definitions-gc_contours), the number of points is assumed to be the number of points previously pointed.*

1. *Finalize the loading:* Once the scapula is fully pointed, we perform ["Step 3"](#step-3-scapulafrom_reference_scapula) for this reference scapula. Then, we perform the loading of all the statistical and EOS scapula geometries, i.e., ["Step 2.a"](#step-2a-constructs-the-statistical-scapulas) or ["Step 2.b"](#step-2b-constructs-the-eos-scapulas), respectively.

#### ***Step 2.a***: Generate the statistical scapula geometries

    In brief, construct the statistical scapula geometries from the average scapula.

    Main function called: 
        Scapula.generator(number_to_generate, model, models_folder, reference)
    where 
        number_to_generate, The number of scapulas to generate;
        model, The type of scapula to generate (i.e. "A" for asymptomatic or "P" for pathological);
        models_folder, The folder where the average scapula geometries are stored;
        reference, is the scapula loaded in Step 1.
    
    Output of the Step 2.a:
        The scapula geometry
        Position of bony landmarks on that geometry

1. *Generate geometries:* From two statistical models, a total of 1000 scapula geometries were generated, i.e., 500 asymptomatic (A) and 500 pathological (P). The A models were constructed with a range of 7, and the P models were constructed with a range of 8. *Note: I do not go into the details of the construction of the models here as they rely on the example script provided by the Hagemeister team.*
   
2. *Point the landmarks:* Since the index of the vertices of the scapula geometries and those from the reference scapula are shared by definition, we use the indices of the landmarks of the reference scapula and apply them to the scapula geometry to locate the landmarks.

3. *Finalize the loading:* Once the scapula is fully pointed, we perform the ["Step 3"](#step-3-scapulafrom_reference_scapula) for all the scapula.

#### ***Step 2.b***: Load the EOS scapula geometries

    In brief, construct the scapula geometries from an EOS scan.

    Main function called: 
        Scapula.from_reference_scapula(
            geometry, reference, shared_indices_with_reference=True, is_left
        )
    where 
        geometry, Either the path to the scapula geometry file or the scapula geometry itself;
        reference, The scapula loaded in Step 1;
        shared_indices_with_reference, A boolean (always set to True) indicating if the index of the vertices of the scapula geometry and the reference scapula are shared (i.e. the scapula geometry was morphed from the reference scapula);
        is_left, A boolean indicating if the scapula is from a left-hand side scan.
        
    Output of the Step 2.b:
        The scapula geometry
        Position of bony landmarks on that geometry

    Note: this algorithm exclude the use of the *.stl files as they are not simply morphed from the reference scapula like the *.ply files are. 

1. *Load the scapula geometries:* A total of 80 scans of scapula geometry (28 from asymptomatic scapulas and 52 from pathologic scapulas) were available. For each scapula, we call the `Scapula.from_reference_scapula` method to either load the geometry from an object file (`*.ply`) or directly store from the geometry itself. If the geometry was from a left-hand side scapula, we set the `is_left` parameter to `True`.

2. *Mirror the geometry if needed:* While loading the geometry, if the scapula was marked as left, we mirror the scapula so that it becomes right-hand side by multiplying the x-axis by -1.

3. *Points the landmarks:* Since the index of the vertices of the scapula geometries and those from the reference scapula are shared (EOS morphs a reference geometry to conform to the scan), we use the indices of the landmarks of the reference scapula and apply them to the scapula geometry to locate the landmarks.

4. *Finalize the loading:* Once the scapula is fully pointed, we perform the ["Step 3"](#step-3-scapulafrom_reference_scapula) for all the scapula.

#### ***Step 3***: Construct the joint coordinate systems (JCS)

    In brief, internal step that computes extra landmarks and defines all the JCS used in the literature.

1. *Compute extra landmarks:* from the pointed bony landmarks, we compute some extra landmarks as such:
   1. [*GC_MID*](#definitions-gc_mid) as the mean of [*IE*](#definitions-ie) and [*SE*](#definitions-se);
   2. [*GC_NORMAL*](#definitions-gc_normal) as the normal of the best fitting plane of the [*GC_CONTOURS*](#definitions-gc_contours). To ensure the normal is pointing outwards, we project the normal in the [*ISB*](#definitions-isb) reference system and flip it if the Z-component is negative;
   3. [*GC_CIRCLE_CENTER*](#definitions-gc_circle_center) as the center of the best fitting circle of the [*GC_CONTOURS*](#definitions-gc_contours), projected on the best fitting plane, using a least square method;
   4. [*GC_ELLIPSE_CENTER*](#definitions-gc_ellipse_center) as the center of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours), projected on the best fitting plane, using an optimization method;
   5. [*GC_ELLIPSE_MAJOR*](#definitions-gc_ellipse_major) as the major axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours), projected on the best fitting plane, using an optimization method;
   6. [*GC_ELLIPSE_MINOR*](#definitions-gc_ellipse_minor) as the minor axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours), projected on the best fitting plane, using an optimization method.

2. <a id="method-step_3_normalize"></a>*Normalize the data:* Compute the [normalized data](#definitions-normalized_data) from the [raw data](#definitions-raw_data) by homogeneously dividing the geometry coordinates by the norm of the vector [*AI*](#definitions-ai) to [*AA*](#definitions-aa), and then by transporting the geometry so the global reference frame coincides with the [*ISB*](#definitions-isb) joint coordinate system.
   
3. *Construct the [*JCS*](#definitions-jcs):* From the [*normalized data*](#definitions-normalized_data), we construct all the JCS required.


#### ***Step 4***: Average the JCS

    In brief, compute the average transformation matrix from the ISB to all JCS

    Main functions called: 
        all_rt = Scapula.get_frame_of_reference(scapulas, target_reference_system)
        average_rts = MatrixHelpers.average_matrices(all_rt)
    where 
        scapulas, A collection of the scapulas outputed from the Step 3;
        target_reference_system, The JCS to average;
        all_rt, A list of all the transformation matrices from ISB to target JCS for all the scapulas;

    Output of the Step 4:
        all_rt, A list of all the transformation matrices from ISB to target JCS for all the scapulas;
        average_rts, The average transformation matrix from ISB to target JCS across all the scapulas.

    Note: this step is performed separately for the statistical and EOS scapulas.

1. *Compute the average rotation matrix:* From the JCS of all the scapulas, we extract the rotation part (i.e. the 3x3 matrix) and average them using the following algorithm:
   1. Construct a non-orthogonal matrix by taking the arithmetic mean of each element of the rotation matrices (i.e. the 3x3 matrices from the top-left corner of the JCS);
   2. Compute the singular value decomposition of that non-orthogonal matrix;
   3. Construct the average rotation matrix by multiplying the U and V' matrices of the singular value decomposition.

2. *Compute the average translation vector:* From the JCS of all the scapulas, we extract the translation part (i.e. the 3x1 vector) and average them using the following algorithm:
   1. Extract the translation vectors, i.e., the rightmost column of the JCS matrices;
   2. Multiply the translation vector by the scaling factor computed at [*Step 3.2*](#method-step_3_normalize) to get the real-world translation vector;
   3. Compute the arithmetic mean of each element of the translation vectors;

#### ***Step 5***: Compute errors and standard deviations

    In brief, compute the orientation and translation errors between the JCS of all the scapulas and the average JCS.

    Main functions called: 
        rotation_errors = MatrixHelpers.angle_between_rotations(all_rt, average_rt)
        translation_errors = MatrixHelpers.distance_between_origins(all_rt, average_rt)
    where 
        all_rt and average_rt, are the output of the Step 4;

    Output of the Step 5:
        rotation_errors, The orientation errors and standard deviations across all the scapulas;
        translation_errors, The translation errors and standard deviations across all the scapulas.

    Note: As for the Step 4, this step is performed separately for the statistical and EOS scapulas.

1. *Compute the orientation errors:* From the JCS of all the scapulas and the average JCS, we compute the orientation errors using the following algorithm:
   1. Get the minimal angle between the rotation matrices of the JCS (R1) and the average JCS (R2) using the formula: `angle = arccos((trace(R1'*R2)-1)/2)`;
   2. Compute the mean and standard deviation of the angles.

2. *Compute the translation errors:* From the JCS of all the scapulas and the average JCS, we compute the translation errors using the following algorithm:
   1. Get the euclidian distance between the origins of the JCS (T1) and the average JCS (T2) using the formula: `distance = norm(T1-T2)`;
   2. Compute the mean and standard deviation of the distances.
