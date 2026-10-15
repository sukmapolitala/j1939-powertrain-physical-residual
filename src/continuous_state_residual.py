from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer

@dataclass
class ContinuousStateResidualDetector:
    context_features: list[str]
    response_features: list[str]
    degree: int = 2
    calibration_quantile: float = 0.99
    ewma_alpha: float = 0.25
    persistence_samples: int = 3
    residual_z_clip: float = 8.0

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame):
        self.ximp = SimpleImputer(strategy="median")
        self.yimp = SimpleImputer(strategy="median")
        self.xscale = RobustScaler()
        self.yscale = RobustScaler()

        X = self.xscale.fit_transform(self.ximp.fit_transform(train[self.context_features]))
        Y = self.yscale.fit_transform(self.yimp.fit_transform(train[self.response_features]))
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        Z = self.poly.fit_transform(X)

        self.models=[]
        for j in range(Y.shape[1]):
            m=HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=1000)
            m.fit(Z,Y[:,j])
            self.models.append(m)

        R=Y-np.column_stack([m.predict(Z) for m in self.models])
        self.residual_median=np.nanmedian(R,axis=0)
        mad=np.nanmedian(np.abs(R-self.residual_median),axis=0)
        self.residual_scale=np.maximum(1.4826*mad,1e-5)
        W=np.clip((R-self.residual_median)/self.residual_scale,-self.residual_z_clip,self.residual_z_clip)
        cov=LedoitWolf().fit(W).covariance_
        self.residual_cov=cov+np.eye(cov.shape[0])*1e-8

        raw=self.raw_score(calibration)
        self.raw_threshold=float(np.quantile(raw[np.isfinite(raw)],self.calibration_quantile))
        norm=raw/self.raw_threshold
        ew=self._ewma(np.nan_to_num(norm,nan=0.0))
        self.ewma_threshold=float(np.quantile(ew[np.isfinite(ew)],self.calibration_quantile))
        return self

    def _observed_response_mask(self, df):
        masks=[]
        for s in self.response_features:
            m=pd.to_numeric(df[s],errors="coerce").notna().to_numpy()
            vc=f"valid__{s}"
            if vc in df:
                m &= pd.to_numeric(df[vc],errors="coerce").fillna(0).to_numpy()>0
            masks.append(m)
        return np.column_stack(masks)

    def residuals(self,df):
        X=self.xscale.transform(self.ximp.transform(df[self.context_features]))
        # Keep response missingness separate; imputation is only for transformer mechanics.
        obs=self._observed_response_mask(df)
        Y=self.yscale.transform(self.yimp.transform(df[self.response_features]))
        Z=self.poly.transform(X)
        Yhat=np.column_stack([m.predict(Z) for m in self.models])
        R=Y-Yhat
        R[~obs]=np.nan
        return R,obs

    def raw_score(self,df):
        R,obs=self.residuals(df)
        out=np.full(len(R),np.nan)
        for i in range(len(R)):
            idx=np.flatnonzero(obs[i] & np.isfinite(R[i]))
            if not len(idx): continue
            z=(R[i,idx]-self.residual_median[idx])/self.residual_scale[idx]
            z=np.clip(z,-self.residual_z_clip,self.residual_z_clip)
            cov=self.residual_cov[np.ix_(idx,idx)]
            inv=np.linalg.pinv(cov)
            out[i]=np.sqrt(max(0.0,float(z@inv@z.T))/len(idx))
        return out

    def _ewma(self,s):
        s=np.asarray(s,float)
        out=np.empty(len(s),float)
        if not len(s): return out
        out[0]=s[0]
        for i in range(1,len(s)):
            out[i]=self.ewma_alpha*s[i]+(1-self.ewma_alpha)*out[i-1]
        return out

    def predict(self,df):
        raw=self.raw_score(df)
        norm=raw/self.raw_threshold
        ew=self._ewma(np.nan_to_num(norm,nan=0.0))
        exceed=ew>self.ewma_threshold
        alarm=np.zeros(len(ew),dtype=int)
        run=0
        for i,f in enumerate(exceed):
            run=run+1 if f else 0
            alarm[i]=int(run>=self.persistence_samples)
        R,obs=self.residuals(df)
        return {
            "raw_score":raw,
            "normalized_score":norm,
            "ewma_score":ew,
            "alarm":alarm,
            "residuals":R,
            "observed_mask":obs
        }

    def parameter_summary(self):
        return {
            "context_features":self.context_features,
            "response_features":self.response_features,
            "degree":self.degree,
            "calibration_quantile":self.calibration_quantile,
            "ewma_alpha":self.ewma_alpha,
            "persistence_samples":self.persistence_samples,
            "raw_threshold":self.raw_threshold,
            "ewma_threshold":self.ewma_threshold,
            "polynomial_features":list(self.poly.get_feature_names_out(self.context_features)),
            "coefficients":{
                self.response_features[j]:{
                    "intercept":float(m.intercept_),
                    "coef":[float(v) for v in m.coef_]
                } for j,m in enumerate(self.models)
            }
        }
