/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Map.h"
#include <sys/stat.h>
#include<mutex>
#define MAPVERBOSE
#define LOGPATH "/home/peter/REU_Workspace/stereo_sensor_project/orbslam_code/data/"

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

void Map::AddKeyFrame(KeyFrame *pKF){
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if(pKF->mnId>mnMaxKFid)
      mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP){
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP){
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF){
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.erase(pKF);

  // TODO: This only erase the pointer.
  // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs){
  unique_lock<mutex> lock(mMutexMap);
  mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange(){
  unique_lock<mutex> lock(mMutexMap);
  mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx(){
  unique_lock<mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames(){
  unique_lock<mutex> lock(mMutexMap);
  return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints(){
  unique_lock<mutex> lock(mMutexMap);
  return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap(){
  unique_lock<mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap(){
  unique_lock<mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints(){
  unique_lock<mutex> lock(mMutexMap);
  return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid(){
  unique_lock<mutex> lock(mMutexMap);
  return mnMaxKFid;
}

void Map::clear(){
  for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
      delete *sit;

  for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
      delete *sit;

  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpReferenceMapPoints.clear();
  mvpKeyFrameOrigins.clear();
}

// Binary version
// TODO: frameid vs keyframeid

KeyFrame* Map::_ReadKeyFrame(ifstream &f, ORBVocabulary &voc, map<long unsigned int, MapPoint*>* relevantMPs, ORBextractor* orb_ext, const cv::FileStorage &fsSettings) {
  #ifdef MAPVERBOSE
    ofstream fKFs;
    string path = LOGPATH;
    path += "ReadKFs.txt";
    fKFs.open(path);
  #endif
  Frame fr;
  fr.mpORBvocabulary = &voc;
  f.read((char*)&fr.mnId, sizeof(fr.mnId));              // ID

  //cerr << " reading keyfrane id " << fr.mnId << endl;
  f.read((char*)&fr.mTimeStamp, sizeof(fr.mTimeStamp));  // timestamp
  cv::Mat Tcw(4,4,CV_32F);                               // position
  f.read((char*)&Tcw.at<float>(0, 3), sizeof(float));
  f.read((char*)&Tcw.at<float>(1, 3), sizeof(float));
  f.read((char*)&Tcw.at<float>(2, 3), sizeof(float));
  Tcw.at<float>(3,3) = 1.;
  
  cv::Mat Qcw(1,4, CV_32F);                             // orientation
  f.read((char*)&Qcw.at<float>(0, 0), sizeof(float));
  f.read((char*)&Qcw.at<float>(0, 1), sizeof(float));
  f.read((char*)&Qcw.at<float>(0, 2), sizeof(float));
  f.read((char*)&Qcw.at<float>(0, 3), sizeof(float));
  Converter::RmatOfQuat(Tcw, Qcw);

  fr.SetPose(Tcw);
  
  f.read((char*)&fr.N, sizeof(fr.N));                    // nb keypoints
  fr.mvKeys.reserve(fr.N);
  fr.mDescriptors.create(fr.N, 32, CV_8UC1);
  fr.mvpMapPoints = vector<MapPoint*>(fr.N,static_cast<MapPoint*>(NULL));
  #ifdef MAPVERBOSE
    fKFs << "Keyframe ID: " << fr.mnId << endl;
  #endif
  for (int i=0; i<fr.N; i++) {
    cv::KeyPoint kp;
    f.read((char*)&kp.pt.x,     sizeof(kp.pt.x));
    f.read((char*)&kp.pt.y,     sizeof(kp.pt.y));
    f.read((char*)&kp.size,     sizeof(kp.size));
    f.read((char*)&kp.angle,    sizeof(kp.angle));
    f.read((char*)&kp.response, sizeof(kp.response));
    f.read((char*)&kp.octave,   sizeof(kp.octave));
    fr.mvKeys.push_back(kp);
    for (int j=0; j<32; j++)
      f.read((char*)&fr.mDescriptors.at<unsigned char>(i, j), sizeof(char));
    unsigned long int mpID;
    f.read((char*)&mpID, sizeof(mpID));
    if (mpID == ULONG_MAX) fr.mvpMapPoints[i] = NULL;
    else fr.mvpMapPoints[i] = (*relevantMPs)[mpID];
    #ifdef MAPVERBOSE
      fKFs << "KP " << i << " MPID " << mpID << endl;
    #endif
  }

  // ADDING FOR TESTING - PETER
  fr.fx = fsSettings["Camera.fx"];
  fr.fy = fsSettings["Camera.fy"];
  fr.cx = fsSettings["Camera.cx"];
  fr.cy = fsSettings["Camera.cy"];
  fr.invfx = 1.0f/fr.fx;
  fr.invfy = 1.0f/fr.fy;
  fr.mDistCoef = cv::Mat(4,1,CV_32F);
  fr.mDistCoef.at<float>(0) = fsSettings["Camera.k1"];
  fr.mDistCoef.at<float>(1) = fsSettings["Camera.k2"];
  fr.mDistCoef.at<float>(2) = fsSettings["Camera.p1"];
  fr.mDistCoef.at<float>(3) = fsSettings["Camera.p2"];
  fr.mK = cv::Mat::eye(3,3,CV_32F);
  fr.mK.at<float>(0,0) = fr.fx;
  fr.mK.at<float>(1,1) = fr.fy;
  fr.mK.at<float>(0,2) = fr.cx;
  fr.mK.at<float>(1,2) = fr.cy;
  //END TESTING


  // mono only for now
  fr.mvuRight = vector<float>(fr.N,-1);
  fr.mvDepth = vector<float>(fr.N,-1);
  fr.mpORBextractorLeft = orb_ext;
  fr.InitializeScaleLevels();

  fr.UndistortKeyPoints();

  fr.AssignFeaturesToGrid();

  fr.ComputeBoW();

  KeyFrame* kf = new KeyFrame(fr, this, NULL);
  kf->mnId = fr.mnId; // bleeee why? do I store that?
  
  // get the number of map points associated with this KFs
  // long unsigned int nKFMPs;
  // f.read((char*)&nKFMPs, sizeof(nKFMPs));
  // for(int i=0; i<nKFMPs; i++){
  //   unsigned long int mpID, mpidx;
  //   f.read((char*)&mpID, sizeof(mpID));
  //   kf->AddMapPoint((*relevantMPs)[mpID], i);
  // }
  for (int i=0; i<fr.N; i++) {
    if (fr.mvpMapPoints[i]) {
      fr.mvpMapPoints[i]->AddObservation(kf, i);
      if (!fr.mvpMapPoints[i]->GetReferenceKeyFrame()) fr.mvpMapPoints[i]->SetReferenceKeyFrame(kf);
    }
  }

  #ifdef MAPVERBOSE
    fKFs.close();
  #endif

  return kf;
}

MapPoint* Map::_ReadMapPoint(ifstream &f) {
  long unsigned int id; 
  f.read((char*)&id, sizeof(id));              // ID
  cv::Mat wp(3,1, CV_32F);
  f.read((char*)&wp.at<float>(0), sizeof(float));
  f.read((char*)&wp.at<float>(1), sizeof(float));
  f.read((char*)&wp.at<float>(2), sizeof(float));
  long int mnFirstKFid=0, mnFirstFrame=0;
  MapPoint* mp = new MapPoint(wp, mnFirstKFid, mnFirstFrame, this);
  mp->mnId = id;
  return mp;
}

void Map::_WriteMapPoint(ofstream &f, MapPoint* mp) {
  f.write((char*)&mp->mnId, sizeof(mp->mnId));               // id: long unsigned int
  cv::Mat wp = mp->GetWorldPos();
  f.write((char*)&wp.at<float>(0), sizeof(float));           // pos x: float
  f.write((char*)&wp.at<float>(1), sizeof(float));           // pos y: float
  f.write((char*)&wp.at<float>(2), sizeof(float));           // pos z: float
}

void Map::_WriteKeyFrame(ofstream &f, KeyFrame* kf) {
  #ifdef MAPVERBOSE
    ofstream fKFs;
    string path = LOGPATH;
    path += "WriteKFs.txt";
    fKFs.open(path);
  #endif
  f.write((char*)&kf->mnId, sizeof(kf->mnId));                 // id: long unsigned int
  f.write((char*)&kf->mTimeStamp, sizeof(kf->mTimeStamp));     // ts: double

  #if 0
    cerr << "wrORB_SLAM2::MapPoint* mp = _ReadMapPoint(f);itting keyframe id " << kf->mnId << " ts " << kf->mTimeStamp << " frameid " << kf->mnFrameId << " TrackReferenceForFrame " << kf->mnTrackReferenceForFrame << endl;
    cerr << " parent " << kf->GetParent() << endl;
    cerr << "children: ";
    for(auto ch: kf->GetChilds())
      cerr << " " << ch->mnId;
    cerr <<endl;
    cerr << kf->mnId << " connected: (" << kf->GetConnectedKeyFrames().size() << ") ";
    for (auto ckf: kf->GetConnectedKeyFrames())
      cerr << ckf->mnId << "," << kf->GetWeight(ckf) << " ";
    cerr << endl;
  #endif
  
  cv::Mat Tcw = kf->GetPose();
  f.write((char*)&Tcw.at<float>(0,3), sizeof(float));          // px: float
  f.write((char*)&Tcw.at<float>(1,3), sizeof(float));          // py: float
  f.write((char*)&Tcw.at<float>(2,3), sizeof(float));          // pz: float
  
  vector<float> Qcw = Converter::toQuaternion(Tcw.rowRange(0,3).colRange(0,3));
  f.write((char*)&Qcw[0], sizeof(float));                      // qx: float
  f.write((char*)&Qcw[1], sizeof(float));                      // qy: float
  f.write((char*)&Qcw[2], sizeof(float));                      // qz: float
  f.write((char*)&Qcw[3], sizeof(float));                      // qw: float
  
  f.write((char*)&kf->N, sizeof(kf->N));                       // nb_features: int
  #ifdef MAPVERBOSE
    fKFs << "Keyframe ID: " << kf->mnId << endl;
  #endif
  for (int i=0; i<kf->N; i++) {
    cv::KeyPoint kp = kf->mvKeys[i];
    f.write((char*)&kp.pt.x,     sizeof(kp.pt.x));               // float
    f.write((char*)&kp.pt.y,     sizeof(kp.pt.y));               // float
    f.write((char*)&kp.size,     sizeof(kp.size));               // float
    f.write((char*)&kp.angle,    sizeof(kp.angle));              // float
    f.write((char*)&kp.response, sizeof(kp.response));           // float
    f.write((char*)&kp.octave,   sizeof(kp.octave));             // int
    for (int j=0; j<32; j++) 
      f.write((char*)&kf->mDescriptors.at<unsigned char>(i,j), sizeof(char));
    unsigned long int mpID;
    MapPoint* mp = kf->GetMapPoint(i);
    if (mp == NULL) mpID = ULONG_MAX;
    else mpID = mp->mnId;
    f.write((char*)&mpID, sizeof(mpID));
    #ifdef MAPVERBOSE
      fKFs << "KP " << i << " MPID " << mpID << endl;
    #endif
  }

  #ifdef MAPVERBOSE
    fKFs.close();
  #endif
}

bool Map::Save(const string &filename) {
  cerr << "Map: Saving to " << filename << endl;
  ofstream f;
  f.open(filename.c_str(), ios_base::out|ios::binary);

  #ifdef MAPVERBOSE
    cerr << "RUNNING IN VERBOSE MODE" <<endl;
    cerr << "Writing log to: " << filename << ".txt" << endl;
    ofstream ftext;
    ftext.open(filename+".txt");
    ftext << "MAP SAVING LOG" << endl;
  #endif
  
  // build a list of mspKeyFrames and the connected keyframes for each (needed for localization)
  // use of map ensures uniqueness and ordering which makes finding simple
  map<long unsigned int, KeyFrame*> relevantKFs;
  for(auto kf:mspKeyFrames){ // keyframes from mspKeyFrames
    relevantKFs.insert({kf->mnId, kf});
    for(auto ckf:kf->GetConnectedKeyFrames()){ // and those referenced by them
      relevantKFs.insert({ckf->mnId, ckf});
    }
  }

  // build a list of mappoints
  map<long unsigned int, MapPoint*> relevantMPs;
  for(auto kfp : relevantKFs){
    set<MapPoint*> mps = kfp.second->GetMapPoints();
    for(auto mp: mps){
      relevantMPs.insert({mp->mnId, mp});
    }
  }
  for(auto mp : mspMapPoints)
    relevantMPs.insert({mp->mnId, mp});

  // store the number of mappoints
  long unsigned int nRelevantMPs = relevantMPs.size();
  cerr << "  writing " << nRelevantMPs << " mappoints" << endl;
  f.write((char*)&nRelevantMPs, sizeof(nRelevantMPs));
  // store the mappoints
  for(auto mpp: relevantMPs)
    _WriteMapPoint(f, mpp.second);

  #ifdef MAPVERBOSE
    ftext << "MAP POINTS:" << endl;
    ftext << "Should be: " << relevantMPs.size() << " relevant map points" << endl;
    ftext << "Are: " << relevantMPs.size() << " in map" << endl;
    for(auto mpp : relevantMPs){
      ftext << "ID: " << mpp.second->mnId << endl;
    }
    ftext << "--------------------------------------" << endl;
  #endif

  // store the number of mspMapPoints
  long unsigned int nMspMapPoints = mspMapPoints.size();
  f.write((char*)&nMspMapPoints, sizeof(nMspMapPoints));
  // store the ids of the mspMapPoints
  for(auto mp : mspMapPoints){
    long unsigned int mpID = mp->mnId;
    f.write((char*)&mpID, sizeof(mpID));
  }

  #ifdef MAPVERBOSE
    ftext << "Should be: " << nMspMapPoints << " MSPMapPoints" << endl;
    ftext << "Are: " << mspMapPoints.size() << " in map " << endl;
    for(auto mp : mspMapPoints){
      if(mp) ftext << "ID: " << mp->mnId << endl;
      else ftext << "ID: " << " NULL" << endl;
    }
    ftext << "--------------------------------------" << endl; 
  #endif

  //store the number of keyframes
  long unsigned int nRelevantKFs = relevantKFs.size();
  cerr << "  writing " << nRelevantKFs << " keyframes" << endl;
  f.write((char*)&nRelevantKFs, sizeof(nRelevantKFs));
  // store the keyframes
  for(auto kfp : relevantKFs)
    _WriteKeyFrame(f, kfp.second);

  #ifdef MAPVERBOSE
    ftext << "Should be: " << nRelevantKFs << " Relevant KFs" << endl;
    ftext << "Are " << relevantKFs.size() << " in map" << endl;
    for(auto kfp : relevantKFs)
      ftext << "ID: " << kfp.second->mnId << endl;
    ftext << "--------------------------------------" << endl;
  #endif

  // store the number of mspKeyFrames
  long unsigned int nMspKeyFrames = mspKeyFrames.size();
  f.write((char*)&nMspKeyFrames, sizeof(nMspKeyFrames));
  // store the ids of the mspKeyFrames
  for(auto kf : mspKeyFrames){
    long unsigned int kfID = kf->mnId;
    f.write((char*)&kfID, sizeof(kfID));
  }

  #ifdef MAPVERBOSE
    ftext << "Should be: " << mspKeyFrames.size() << " MSPKeyFrames" << endl;
    ftext << "Are: " << mspKeyFrames.size() << " in map" << endl;
    for(auto kf : mspKeyFrames)
      ftext << "ID: " << kf->mnId << endl;
    ftext << "--------------------------------------" << endl; 
  #endif

  // store tree and graph
  #ifdef MAPVERBOSE
    ftext << "TREE AND GRAPH" << endl;
  #endif
  for(auto kfp : relevantKFs) {
    KeyFrame* parent = kfp.second->GetParent();
    long unsigned int parent_id = ULONG_MAX; 
    if (parent) parent_id = parent->mnId;
    f.write((char*)&parent_id, sizeof(parent_id));
    long unsigned int nb_con = kfp.second->GetConnectedKeyFrames().size();
    f.write((char*)&nb_con, sizeof(nb_con));
    #ifdef MAPVERBOSE
      ftext << "KF ID: " << kfp.second->mnId << endl;
      ftext << "    KFP ID: " << parent_id << endl;
      ftext << "    Connected KFs: (" << nb_con << ")" << endl;
    #endif
    for (auto ckf: kfp.second->GetConnectedKeyFrames()) {
      int weight = kfp.second->GetWeight(ckf);
      f.write((char*)&ckf->mnId, sizeof(ckf->mnId));
      f.write((char*)&weight, sizeof(weight));
      #ifdef MAPVERBOSE
        ftext << "    ID: " << ckf->mnId << " Weight: " << weight << endl;
      #endif
    }
  }

  f.close();
  cerr << "Map: finished saving" << endl;
  struct stat st;
  stat(filename.c_str(), &st);
  cerr << "Map: saved " << st.st_size << " bytes" << endl;

  #ifdef MAPVERBOSE
    ftext.close();
  #endif

  return true;
}

bool Map::Load(const string &filename, ORBVocabulary &voc, const string &strSettingsFile) {
  // Load ORB parameters from the setings file
  cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
  int nFeatures = fsSettings["ORBextractor.nFeatures"];
  float scaleFactor = fsSettings["ORBextractor.scaleFactor"];
  int nLevels = fsSettings["ORBextractor.nLevels"];
  int fIniThFAST = fsSettings["ORBextractor.iniThFAST"];
  int fMinThFAST = fsSettings["ORBextractor.minThFAST"];
  ORB_SLAM2::ORBextractor orb_ext = ORB_SLAM2::ORBextractor(nFeatures, scaleFactor, nLevels, fIniThFAST, fMinThFAST);

  // open map file
  cerr << "Map: reading from " << filename << endl;
  ifstream f;
  f.open(filename.c_str());

  #ifdef MAPVERBOSE
    cerr << "RUNNING IN VERBOSE MODE" <<endl;
    cerr << "Writing log to: " << filename << "LOADED.txt" << endl;
    ofstream ftext;
    ftext.open(filename+"LOADED.txt");
    ftext << "MAP LOADING LOG" << endl;
  #endif

  // Determine the number of map point to load
  long unsigned int nMPs, max_id=0;
  f.read((char*)&nMPs, sizeof(nMPs));
  cerr << "reading " << nMPs << " mappoints" << endl;

  // load in the map points
  map<long unsigned int, MapPoint*> relevantMPs;
  for(long unsigned int i=0; i<nMPs; i++){
    ORB_SLAM2::MapPoint* mp = _ReadMapPoint(f);
    if (mp->mnId>=max_id) max_id=mp->mnId;
    relevantMPs.insert({mp->mnId, mp});
  }
  ORB_SLAM2::MapPoint::nNextId = max_id+1; // that is probably wrong if last mappoint is not here :(

  #ifdef MAPVERBOSE
    ftext << "MAP POINTS:" << endl;
        ftext << "Should be: " << nMPs << " relevant map points" << endl;
    ftext << "Are: " << relevantMPs.size() << " in map" << endl;
    for(auto mpp : relevantMPs){
      ftext << "ID: " << mpp.second->mnId << endl;
    }
    ftext << "--------------------------------------" << endl;
  #endif

  // read the number of mspMapPoints
  long unsigned int nMspMapPoints;
  f.read((char*)&nMspMapPoints, sizeof(nMspMapPoints));
  // read the ids of the mspMapPoints and add them to the list
  for(long unsigned int i=0; i<nMspMapPoints; i++){
    long unsigned int mpID;
    f.read((char*)&mpID, sizeof(mpID));
    AddMapPoint(relevantMPs[mpID]);
  }

  #ifdef MAPVERBOSE
    ftext << "Should be: " << nMspMapPoints << " MSPMapPoints" << endl;
    ftext << "Are: " << mspMapPoints.size() << " in map " << endl;
    for(auto mp : mspMapPoints){
      if(mp) ftext << "ID: " << mp->mnId << endl;
      else ftext << "ID: " << " NULL" << endl;
    }
    ftext << "--------------------------------------" << endl;
  #endif

  //Determine the number of keyframes to load
  long unsigned int nKFs;
  f.read((char*)&nKFs, sizeof(nKFs));
  cerr << "reading " << nKFs << " keyframe" << endl; 
  // Load the keyframes
  map<long unsigned int, KeyFrame*> relevantKFs;
  for (long unsigned int i=0; i<nKFs; i++){ 
    KeyFrame* kf = _ReadKeyFrame(f, voc, &relevantMPs, &orb_ext, fsSettings); 
    relevantKFs.insert({kf->mnId, kf});
  }

  #ifdef MAPVERBOSE
    ftext << "Should be: " << nKFs << " Relevant KFs" << endl;
    ftext << "Are " << relevantKFs.size() << " in map" << endl;
    for(auto kfp : relevantKFs)
      ftext << "ID: " << kfp.second->mnId << endl;
    ftext << "--------------------------------------" << endl;
  #endif

  // read the number of mspKeyFrames
  long unsigned int nMspKeyFrames;
  f.read((char*)&nMspKeyFrames, sizeof(nMspKeyFrames));
  // read the ids of the mspKeyFrames and add them to the list
  for(long unsigned int i=0; i<nMspKeyFrames; i++){
    long unsigned int kfID;
    f.read((char*)&kfID, sizeof(kfID));
    AddKeyFrame(relevantKFs[kfID]);
  }

  #ifdef MAPVERBOSE
    ftext << "Should be: " << nMspKeyFrames << " MSPKeyFrames" << endl;
    ftext << "Are: " << mspKeyFrames.size() << " in map" << endl;
    for(auto kf : mspKeyFrames)
      ftext << "ID: " << kf->mnId << endl;
    ftext << "--------------------------------------" << endl; 
  #endif

  // Load Spanning tree
  #ifdef MAPVERBOSE
    ftext << "TREE AND GRAPH" << endl;
  #endif
  for(auto kfp : relevantKFs) {
    long unsigned int parent_id;
    f.read((char*)&parent_id, sizeof(parent_id));          // parent id
    if (parent_id != ULONG_MAX)
      kfp.second->ChangeParent(relevantKFs[parent_id]);
    long unsigned int nb_con;                             // number connected keyframe
    f.read((char*)&nb_con, sizeof(nb_con)); 
    #ifdef MAPVERBOSE
      ftext << "KF ID: " << kfp.second->mnId << endl;
      ftext << "    KFP ID: " << parent_id << endl;
      ftext << "    Connected KFs: (" << nb_con << ")" << endl;
    #endif 
    for (long unsigned int i=0; i<nb_con; i++) {
      long unsigned int id; int weight;
      f.read((char*)&id, sizeof(id));                   // connected keyframe
      f.read((char*)&weight, sizeof(weight));           // connection weight
      kfp.second->AddConnection(relevantKFs[id], weight);
      #ifdef MAPVERBOSE
        ftext << "    ID: " << id << " Weight: " << weight << endl;
      #endif
    }
  }
  // MapPoints descriptors
  for(auto mpp : relevantMPs) {
    if(mpp.second && mpp.first!=ULONG_MAX){
      mpp.second->ComputeDistinctiveDescriptors();
      mpp.second->UpdateNormalAndDepth();
    }
  }

  #ifdef MAPVERBOSE
    // ftext << "There are " << mspKeyFrames.size() << " MSPKeyFrames" << endl;
    // for(auto kf:mspKeyFrames){ // keyframes from mspKeyFrames
    //     ftext << "MSP KF: " << kf->mnId << endl;
    //     ftext << "Connected KeyFrames:" << endl;
    //   for(auto ckf:kf->GetConnectedKeyFrames()){ // and those referenced by them
    //     ftext << "ID: " << ckf->mnId << endl;
    //   }
    // }

    //ftext << "Read " << relevantKFs.size() << " relevant key frames" << endl;
    //ftext << "--------------------------------------" << endl;
    // ftext << "There are " << mspMapPoints.size() << " MSPMapPoints" << endl;
    
    // ftext << "--------------------------------------" << endl;
    // ftext << "Read " << relevantKFs.size() << " relevant map points" << endl;
    // ftext << "Relevant Key Frames:" << endl;
    // for(auto kfp : relevantKFs){
    //   ftext << "ID: " << kfp.second->mnId << endl;
    // }

    ftext.close();
  #endif

  f.close();
  cerr << "Map: finished Loading" << endl;

  return true;
}

} //namespace ORB_SLAM
