#printing the start methods 
echo "Starting the data download for CLICICAL RAG"  

#creating a directory
echo "Creating data Directory"
mkdir -p data/raw/statpearls
mkdir -p data/processed/statpearls

#Downloading the Data from the Url
echo "Downloading the Data from the Url"
wget -O data/raw/statpearls/statpearls_NBK430685.tar.gz https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz

#Extract the Data form the url
echo "Extracting the data to data/statpearls"
tar -xzvf data/raw/statpearls/statpearls_NBK430685.tar.gz -C data/raw/statpearls/

#Now verfiying the number of files 
echo "Verifying the number of files"
FILE_COUNT=$(ls data/raw/statpearls/*.nxml 2>/dev/null | wc -l)
echo "Total .nxml files downloaded: $FILE_COUNT"

echo "Dataset size:"
du -sh data/raw/statpearls/ 
# ── Step 7: Test APIs ──
echo ""
echo "Testing RxNorm API..."
curl -s "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=metformin" | python3 -m json.tool | head -10

echo ""
echo "Testing FDA Drug Label API..."
curl -s "https://api.fda.gov/drug/label.json?search=metformin&limit=1" | python3 -m json.tool | head -10

echo ""
echo "Download complete. Ready to build."
