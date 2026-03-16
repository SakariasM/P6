#!/bin/bash
#SBATCH --job-name=check_containers
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/ceph/home/aau/%u/P6/logs/check_containers_%j.out
#SBATCH --error=/ceph/home/aau/%u/P6/logs/check_containers_%j.err

echo "=== Checking for containers ==="
echo ""

echo "1. Checking /ceph/container/"
ls -lah /ceph/container/ 2>&1 || echo "Directory not accessible"

echo ""
echo "2. Searching for .sif files in /ceph/"
find /ceph -name "*.sif" -type f 2>/dev/null | head -20

echo ""
echo "3. Checking common locations:"
ls -lh /ceph/container/*.sif 2>&1 || echo "No .sif in /ceph/container/"
ls -lh ~/containers/*.sif 2>&1 || echo "No .sif in ~/containers/"
ls -lh /ceph/project/*/containers/*.sif 2>&1 || echo "No .sif in project containers/"

echo ""
echo "4. Checking if Python is available without container:"
which python python3 2>&1 || echo "No python found"
python --version 2>&1 || echo "Python not executable"
python3 --version 2>&1 || echo "Python3 not executable"

echo ""
echo "5. Environment info:"
echo "USER: $USER"
echo "HOME: $HOME"
echo "PWD: $(pwd)"
