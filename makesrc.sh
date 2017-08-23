mkdir svm_smo
cp -r src/* svm_smo/
tar -cvz --exclude='.gitignore' --exclude='svmsmo.tar.gz' --exclude='.git' --exclude='junk' --exclude='*.asv' --exclude='*.png' --exclude='._*' --exclude='*.dll' --exclude='*.mexa64' --exclude='*.mexw64' --exclude='*.so' --exclude='*.mexmaci64' --exclude='*.dylib' -f submission/svmsmo_1.tar.gz svm_smo
rm -rf svm_smo

