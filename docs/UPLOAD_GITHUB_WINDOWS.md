# Create the GitHub repository from Windows

1. Sign in to GitHub as **thoothfairy**, then create an **empty** repository named `j1939-powertrain-physical-residual`.
2. Select **Public** only after reviewing repository rights and code contents. Do not select GitHub auto-generated README, license or gitignore because these are supplied in the ZIP.
3. Extract `J1939_Powertrain_Physical_Residual_PUBLIC_REPO.zip` to a local folder such as `C:\Projects\j1939-powertrain-physical-residual`.
4. Open PowerShell **inside** that folder and run:

```powershell
git init
git branch -M main
git add .
git status --short
git commit -m "Initial reproducible J1939 physical-residual research release"
git remote add origin https://github.com/thoothfairy/j1939-powertrain-physical-residual.git
git push -u origin main
```

If your global git identity is already configured, the commit uses your own configured identity. If Git prompts you for authentication, sign in with the browser flow / Git Credential Manager; never add a personal access token to the code or README.

To check that no raw/derived traces are included before pushing:

```powershell
git ls-files | Select-String -Pattern "local_raw|local_derived|primary_predictions|models/"
```

**Expected:** no matches.

Do not claim the GitHub repository URL is active until it has actually been created and opened successfully.
