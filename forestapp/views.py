from django.shortcuts import render
import os
import requests
from datetime import datetime
from django.http import FileResponse
from django.http import JsonResponse
from django.http import HttpResponse 
directory_path = 'forestapp\\new models'
file_names = os.listdir(directory_path)

# Extract timestamps from file names
timestamps = [datetime.strptime(file_name.split('_')[1], '%Y%m%d%H%M%S') for file_name in file_names]

# Identify the file with the latest timestamp
latest_file_index = timestamps.index(max(timestamps))
latest_file = file_names[latest_file_index]
print(latest_file,2)

def version(request):
    if request.method=='GET':
        content={"version":latest_file}
        return JsonResponse(content)
    

def download(request):
    if request.method=='GET':
        file_path = f'forestapp\\new models\\{latest_file}'
        print(file_path)
        if os.path.exists(file_path):
            # Open the file in binary read mode
            file = open(file_path, 'rb')
            
            # Create a FileResponse with the file content
            response = FileResponse(file)
            
            # Set the Content-Disposition header for downloading
            response['Content-Disposition'] = f'attachment; filename={latest_file}'
            
            return response
        else:
            return HttpResponse('File not found')
