import requests
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def insert_url():
    url = input("Link- ")
    try:
        response = requests.get(url)
        print("URL valido.")
        return url
    except requests.ConnectionError as exception:
        print("URL invalido.")
        insert_url()

def check_path(url):
    path = input("Pasta- ")
    
    if not path:
        # if path isn't specified, use the domain name of that url as the folder name
        path = urlparse(url).netloc
    if not (os.path.exists(path)):
        os.mkdir(path)
    return path

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_images(url):
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    urls = []
    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            continue
        img_url = urljoin(url, img_url)
        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass
        if is_valid(img_url):
            urls.append(img_url)
    return urls


def download(url, pathname):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    filename = os.path.join(pathname, url.split("/")[-1])

    if(filename.find('.svg') != -1):
        return
    elif(filename.endswith('.jpeg'), filename.endswith('.jpg'), filename.endswith('.png')):
        progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "wb") as f:
            for data in progress.iterable:
                f.write(data)
                progress.update(len(data))
    else:
        print("here")
        return


def main():
    url = insert_url()
    path = check_path(url)
 

    imgs = get_all_images(url)
    for img in imgs:
        download(img, path)
    


if __name__ == "__main__": 
    main()