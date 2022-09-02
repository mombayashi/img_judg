from flickrapi import FlickrAPI
from urllib.request import urlretrieve
import os, time, sys


#　APIキーの取得
key = "a8ec5b10abfe2575345dfea6e3d327d8"
secret = "f56e4f9d66227e87"
wait_time = 1

#　保存フォルダの指定
keyword = sys.argv[1]
savedir = "./" + keyword

#　接続クライアントの作成とサーチの実行
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = keyword,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, license'
)

#　結果の取り出しと格納
photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)