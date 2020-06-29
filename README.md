# Reinforcement Learning for Appendix Localization in Abdominal CT Volume

For localizing appendix, we extended the following work:

W. Abdullah Al and I. D. Yun, "Partial Policy-Based Reinforcement Learning for Anatomical Landmark Localization in 3D Medical Images," in IEEE Transactions on Medical Imaging, vol. 39, no. 4, pp. 1245-1255, April 2020, doi: 10.1109/TMI.2019.2946345.

## Approach summary:
An agent initialized at a random point inside a 3D medical image, moves to a neighborhood point using 6 actions (left-right-up-down-sliceForeward-sliceBackward). By taking an episode of such moves, it attempts to converge to the target landmark.

## Usage
**Example**

`appx_rl.py -dicompath "path/to/dicom/folder" -network_path "net/policy1"`

**Parameters**

`-dicompath` : `"path/to/dicom/folder"`

`-network_path` : `"path/to/net"` : (default: `"net/policy_best"`)

`-init_pos_center`: Center of the sample space for the random initial position. (default: center of the input volume)

`-init_pos_radii`: Radii of the sample space for the random initial position. (default: `5`). Use `0`, for exactly using the center constantly, as the initial position.

`-init_pos_radii_multiplier`: To extend the radii but with stride. (default: `1` for zero stride)

`-max_episode`: number of episodes to explore at each epoch. (default: `5`)

`-max_step`: maximum number of steps per episode. (default: `50`). Keep in mind, the step-size is `2` in the current code.



## Troubleshooting:
**If the localization result is not satisfactory**

- Note that, current average error is about `10mm`
- increasing `max_episode` can improve the result, however, localization time would be increased.







