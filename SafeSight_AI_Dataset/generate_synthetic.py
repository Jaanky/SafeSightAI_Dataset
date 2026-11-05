"""
SafeSight AI - synthetic reading generator (seed fixed for reproducibility).
Synthetic rows are generated from simple per-class statistics estimated from the
14 REAL hand-recorded readings. They are NOT measurements and every synthetic
row is flagged is_synthetic=1 in the CSVs.
"""
import numpy as np, pandas as pd, math, os
rng = np.random.default_rng(20261105)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- REAL readings (transcribed from handwritten field notes) ----
real = [
 # id, session, location_name, class, acc(x,y,z), gyro(x,y,z), flag
 ("SS-R01","A","Outdoor stairs","stairs",(-0.02,9.66,-0.28),(0.056,-0.044,-0.012),""),
 ("SS-R02","A","Entrance","flat_ground",(-0.06,9.57,-0.45),(-0.021,0.003,0.010),""),
 ("SS-R03","A","Ground","flat_ground",(0.09,9.53,0.27),(0.122,-0.062,0.006),""),
 ("SS-R04","A","Stair 1 (near warden)","stairs",(0.29,8.19,4.95),(0.019,-0.11,0.004),"gyro_y may be -0.011; verify"),
 ("SS-R05","A","Near filter 1 (washing area)","washing_area",(0.91,6.36,6.29),(0.083,0.161,0.057),""),
 ("SS-R06","A","Washing area (balcony)","balcony",(-0.35,9.60,1.94),(0.056,-0.063,0.048),""),
 ("SS-R07","A","Outdoor area (hanging clothes)","outdoor_obstacle_area",(-0.24,9.05,3.56),(-0.018,-0.017,0.002),"acc_z read as 3.56 (could be 8.56); verify"),
 ("SS-R08","A","Stair 2 (opp mess up)","stairs",(-0.04,9.13,3.05),(0.62,-0.064,0.002),"gyro_x may be 0.062; verify"),
 ("SS-R09","A","Stair 3 (opp mess down)","stairs",(0.18,7.27,5.85),(0.279,0.036,-0.051),"gyro_z may be -0.054; verify"),
 ("SS-R10","A","Outside mess (cement block)","outdoor_obstacle_area",(-0.51,9.56,1.14),(-0.086,-0.042,0.024),""),
 ("SS-R11","B","Hostel door with chair","door_threshold",(-0.28,9.50,1.26),(-0.014,0.013,0.025),""),
 ("SS-R12","B","Table & door","door_threshold",(-0.16,9.76,0.61),(-0.047,-0.079,-0.017),""),
 ("SS-R13","B","Balcony","balcony",(0.15,9.55,1.19),(0.094,0.014,-0.014),""),
 ("SS-R14","B","Corridor","flat_ground",(-0.17,9.66,-0.42),(None,None,None),"gyroscope not recorded"),
]
hazard = {"flat_ground":"none","stairs":"level_change","washing_area":"wet_surface",
          "balcony":"open_edge","outdoor_obstacle_area":"obstacle_uneven","door_threshold":"threshold_step"}
n_total = {"flat_ground":22,"stairs":30,"washing_area":14,"balcony":18,"outdoor_obstacle_area":18,"door_threshold":18}

def pitch(a): return math.degrees(math.atan2(a[2],a[1]))
mag = lambda v: math.sqrt(sum(x*x for x in v))
rr = pd.DataFrame([{"id":r[0],"cls":r[3],"pitch":pitch(r[4]),"M":mag(r[4]),"x":r[4][0]} for r in real])
M_mu, M_sd = rr.M.mean(), rr.M.std()
# gyro axis scale estimated from real (robust, MAD-based)
g = np.array([v for r in real if r[5][0] is not None for v in r[5] if abs(v)<0.5])
g_scale = 1.4826*np.median(np.abs(g-np.median(g)))

rows=[]
for r in real:
    a,gy=r[4],r[5]
    rows.append(dict(record_id=r[0],source="real",is_synthetic=0,session=r[1],location_class=r[3],
        hazard_type=hazard[r[3]],location_name=r[2],acc_x_ms2=a[0],acc_y_ms2=a[1],acc_z_ms2=a[2],
        gyro_x_rads=gy[0],gyro_y_rads=gy[1],gyro_z_rads=gy[2],transcription_flag=r[6]))
k=1
for cls,n in n_total.items():
    sub=rr[rr.cls==cls]
    n_syn=n-len(sub)
    mu=sub.pitch.mean(); sd=max(sub.pitch.std(ddof=0) if len(sub)>1 else 0, 6.0)   # >=6 deg spread
    xm=sub.x.mean()*0.5
    for _ in range(n_syn):
        M=float(np.clip(rng.normal(M_mu,M_sd),8.9,9.85))
        x=float(np.clip(rng.normal(xm,0.30),-1.0,1.0))
        R=math.sqrt(max(M*M-x*x,0))
        th=math.radians(rng.normal(mu,sd))
        y,z=R*math.cos(th),R*math.sin(th)
        gs=(rng.standard_t(3,3)*g_scale).clip(-0.30,0.30)
        rows.append(dict(record_id=f"SS-S{k:03d}",source="synthetic",is_synthetic=1,session="S",location_class=cls,
            hazard_type=hazard[cls],location_name=f"synthetic_{cls}",acc_x_ms2=round(x,2),acc_y_ms2=round(y,2),
            acc_z_ms2=round(z,2),gyro_x_rads=round(float(gs[0]),3),gyro_y_rads=round(float(gs[1]),3),
            gyro_z_rads=round(float(gs[2]),3),transcription_flag=""))
        k+=1
df=pd.DataFrame(rows)
df["acc_magnitude_ms2"]=np.sqrt(df.acc_x_ms2**2+df.acc_y_ms2**2+df.acc_z_ms2**2).round(3)
df["tilt_pitch_deg"]=np.degrees(np.arctan2(df.acc_z_ms2,df.acc_y_ms2)).round(1)
df["gyro_magnitude_rads"]=np.sqrt(df.gyro_x_rads**2+df.gyro_y_rads**2+df.gyro_z_rads**2).round(3)
cols=["record_id","source","is_synthetic","session","location_class","hazard_type","location_name",
 "acc_x_ms2","acc_y_ms2","acc_z_ms2","acc_magnitude_ms2","tilt_pitch_deg",
 "gyro_x_rads","gyro_y_rads","gyro_z_rads","gyro_magnitude_rads","transcription_flag"]
df=df[cols].sort_values(["is_synthetic","record_id"]).reset_index(drop=True)
# interleave-free: keep real first then synthetic; also shuffled combined order is left to the user
df.to_csv(f"{OUT}/sensor_readings_combined.csv",index=False)
df[df.is_synthetic==0].to_csv(f"{OUT}/sensor_readings_real.csv",index=False)
df[df.is_synthetic==1].to_csv(f"{OUT}/sensor_readings_synthetic.csv",index=False)
summ=df.groupby(["location_class","hazard_type"]).agg(n_total=("record_id","count"),n_real=("is_synthetic",lambda s:(s==0).sum()),
   n_synthetic=("is_synthetic","sum"),acc_y_mean=("acc_y_ms2","mean"),acc_z_mean=("acc_z_ms2","mean"),
   tilt_mean_deg=("tilt_pitch_deg","mean"),tilt_sd_deg=("tilt_pitch_deg","std"),
   gyro_mag_mean=("gyro_magnitude_rads","mean")).round(3).reset_index()
summ.to_csv(f"{OUT}/class_summary.csv",index=False)
print(len(df), df.is_synthetic.value_counts().to_dict())
print(summ.to_string())
print("real acc_mag mean/sd",df[df.is_synthetic==0].acc_magnitude_ms2.agg(["mean","std","min","max"]).round(3).to_dict())
print("syn  acc_mag mean/sd",df[df.is_synthetic==1].acc_magnitude_ms2.agg(["mean","std","min","max"]).round(3).to_dict())
print("real gyro_mag",df[df.is_synthetic==0].gyro_magnitude_rads.agg(["mean","median","max"]).round(3).to_dict())
print("syn gyro_mag",df[df.is_synthetic==1].gyro_magnitude_rads.agg(["mean","median","max"]).round(3).to_dict())
print("g_scale",round(g_scale,4))
