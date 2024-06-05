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
- <a id="definitions-gc_contour_normal"></a>**GC_CONTOUR_NORMAL**: Glenoid cavity normal computed from the best fitting plane of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_circle_center"></a>**GC_CIRCLE_CENTER**: Glenoid cavity as the center point of the best fitting circle of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_center"></a>**GC_ELLIPSE_CENTER**: Glenoid cavity as the center point of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_major"></a>**GC_ELLIPSE_MAJOR**: Glenoid cavity as the major axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-gc_ellipse_minor"></a>**GC_ELLIPSE_MINOR**: Glenoid cavity as the minor axis of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours)
- <a id="definitions-ie"></a>**IE**: Inferior Edge of glenoid
- <a id="definitions-se"></a>**SE**: Superior Edge of glenoid
- <a id="definitions-ts"></a>**TS**: Trigonum Spinae

- <a id="definitions-raw_data"></a>**Raw data**: The raw data is value directly extracted from the data file (.ply, .stl, .obj, etc.). 
- <a id="definitions-normalized_data"></a>**Normalized data**: The normalized data is the raw data that has been normalized so that the scapula is shown in a box of roughly 1x1x1.
 
- <a id="definitions-isb"></a>**ISB**: International Society of Biomechanics

### The algorithm (For Statistical scapula)

#### ***Step 1***: Load the reference scapula
1. Load the reference scapula (asymptomatic) from the database (method `Scapula.from_landmarks(geometry, [prepointed_landmarks]`).
   
   1. Load the object scapula file and perform a rough normalization of the [raw data](#definitions-raw_data) so the scapula are shown in a box of about 1x1x1. To do so, the min and max values of the [raw data](#definitions-raw_data) are extracted and the data is normalized by dividing by the length of the min to max vector. 
   *Note: this normalisation is temporary and will be refined later on.*
   
   2. Request the user to point bony landmarks ([*AA*](#definitions-aa), [*AC*](#definitions-ac), [*AI*](#definitions-ai), [*IE*](#definitions-ie), [*SE*](#definitions-se), [*GC_CONTOURS*](#definitions-gc_contours), [*TS*](#definitions-ts)) on the scapula ([raw data](#definitions-raw_data)) on screen. For the [*GC_CONTOURS*](#definitions-gc_contours), the user is also asked to inform how many points should be pointed on the glenoid cavity before starting to actually point them on the scapula. 
   *Note: In order to reduce the developping time, the results were manually saved (and added to the main script) to prevent from having to perform this step each time the script is run. In that case, for the [*GC_CONTOURS*](#definitions-gc_contours), the number of points is assumed to be the number of points previously pointed.*

   3. Compute glenoid values, that is 
      1. [*GC_MID*](#definitions-gc_mid) as the mean of [*IE*](#definitions-ie) and [*SE*](#definitions-se)
      2. [*GC_CIRCLE_CENTER*](#definitions-gc_circle_center) as the center of the best fitting circle of the [*GC_CONTOURS*](#definitions-gc_contours) (projected on the best fitting plane) using a least square method  
      3. [*GC_ELLIPSE_CENTER*](#definitions-gc_ellipse_center) as the center of the best fitting ellipse of the [*GC_CONTOURS*](#definitions-gc_contours) (projected on the best fitting plane) using an opimisation method

   4. Recompute the [normalized data](#definitions-normalized_data) from the [raw data](#definitions-raw_data) using the norm of the vector from the newly computed [*AI*](#definitions-ai) and [*AA*](#definitions-aa) as the scaling homogeneous factor, and transport the [normalized data](#definitions-normalized_data) so that it coincides with the [*ISB*](#definitions-isb) reference system.