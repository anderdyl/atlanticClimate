import os
import requests
import numpy as np

def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


years = np.arange(1880,2022)
months = np.arange(1,13)
for ii in years:
    for hh in months:
        if hh < 10:
            date = str(ii) + "0" + str(hh)
        else:
            date = str(ii) + str(hh)

        file = "ersst.v5." + date + ".nc"
        print(file)
        download(os.path.join("https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/",file), dest_folder="/media/dylananderson/Elements/shusin6_contents/AWT/ERSSTV5/")