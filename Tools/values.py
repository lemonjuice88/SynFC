import tools

url = "https://www.transfermarkt.com.tr/victor-osimhen/marktwertverlauf/spieler/401923"
soup = tools._get_soup(url)

# Yontem A: SvelteKit'in "fetched" verisi
fetched_scripts = soup.find_all("script", attrs={"data-sveltekit-fetched": True})
print(f"Yontem A - data-sveltekit-fetched: {len(fetched_scripts)} tane bulundu")
for s in fetched_scripts:
    print("  URL:", s.get("data-url"))

# Yontem B: TUM script etiketlerini (type filtresi olmadan) say, icinde
# "osimhen" ya da euro isareti gecenleri bul
all_scripts = soup.find_all("script")
print(f"\nYontem B - toplam script sayisi: {len(all_scripts)}")
for i, s in enumerate(all_scripts):
    if s.string and ("osimhen" in s.string.lower() or "€" in s.string):
        print(f"  Script {i} icinde eslesme var, ilk 300 karakter:")
        print(" ", s.string[:300])