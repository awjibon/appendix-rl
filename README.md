# Reinforcement Learning for Appendix Localization in Abdominal CT Volume

For localizing appendix, we extended the following work:

W. Abdullah Al and I. D. Yun, "Partial Policy-Based Reinforcement Learning for Anatomical Landmark Localization in 3D Medical Images," in IEEE Transactions on Medical Imaging, vol. 39, no. 4, pp. 1245-1255, April 2020, doi: 10.1109/TMI.2019.2946345.

## Approach summary:
An agent initialized at a random point inside a 3D medical image, moves to a neighborhood point using 6 actions (left-right-up-down-sliceForeward-sliceBackward). By taking an episode of such moves, it attempts to converge to the target landmark.

## Usage
**Example**

`appx_rl.py -dicom_path "path/to/dicom/folder"`

**Parameters**

`-dicom_path` : `"path/to/dicom/folder"`

`-policy_path` : `"path/to/net"` : (default: `"net"`)

`-max_episode`: number of episodes to explore at each epoch. (default: `5`)

`-max_step`: maximum number of steps per episode. (default: `10`). Note that, the step-size is `2` in the current code.



## Troubleshooting:
**If the localization result is not satisfactory**

- Note that, current average error is about `10mm`
- increasing `max_episode` can improve the result, however, localization time would be increased.







