{-# LANGUAGE TemplateHaskell #-}

import Control.Monad.Trans.Reader
import Control.Monad.Identity
import Control.Monad
import Control.Lens
import Data.List (elemIndex)
import qualified Data.Map as M
import Text.Show.Pretty (ppShow)

-- Given C classes, K resources and a matrix D_{i,j} where the ith,jth entry of D
-- denotes the demand placed by class C_i on resource K_j. Also given a C dimensional
-- vector where C_i denotes the number of users of class i in the system.

type Env a = ReaderT Mva_Parameters Identity a

data Mva_Parameters = Mva_Parameters
  { _demand           :: [[Float]]
  , _resource_count   :: Int
  , _class_count      :: Int
  , _think_times      :: [Float]
  }

makeLenses ''Mva_Parameters

get_demand :: Int -> Int -> Env Float
get_demand i j = do
  current_state <- ask
  let demand_matrix = current_state^.demand
  return $ (demand_matrix !! i) !! j

get_resource_count :: Env Int
get_resource_count = do
  current_state <- ask
  return $ current_state^.resource_count

get_class_count :: Env Int
get_class_count = do
  current_state <- ask
  return $ current_state^.class_count

get_think_time :: Int -> Env Float
get_think_time class_index = do
  current_state <- ask
  let think_time_vector = current_state^.think_times
  return $ think_time_vector !! class_index

decrement_nth :: Int -> [Int] -> [Int]
decrement_nth 0 (x:xs) = (x-1):xs
decrement_nth n (x:xs) = x : (decrement_nth (n-1) xs)

solve_multiclass_mva :: [Int] -> Env (M.Map Int Float, M.Map Int Float, M.Map Int (M.Map Int Float))
solve_multiclass_mva ns
  | all_zeros ns = do
      resource_count <- get_resource_count
      return (empty_q_dict resource_count, M.empty, M.empty)
  | otherwise = do
    class_count <- get_class_count
    resource_count <- get_resource_count
    response_times <- foldM fold_fn M.empty [0..(class_count-1)] 
    throughputs <- foldM (compute_throughput response_times) M.empty [0..(class_count-1)]
    q_lengths <- foldM (compute_q_lengths response_times throughputs) M.empty [0..(resource_count-1)]
    return (q_lengths, throughputs, response_times)

  where
    all_zeros :: [Int] -> Bool
    all_zeros ns = ((length . filter (/=0)) ns) == 0

    empty_q_dict :: Int -> M.Map Int Float
    empty_q_dict k = foldl (\acc v -> M.insert v 0.0 acc) M.empty [0..(k-1)]

    fold_fn
      :: M.Map Int (M.Map Int Float) -> Int -> Env (M.Map Int (M.Map Int Float))
    fold_fn acc c = 
      if ns !! c == 0
        then do
          resource_count <- get_resource_count
          return $ M.insert c (empty_q_dict resource_count) acc
        else do
          let new_vector = decrement_nth c ns
          (q, _, _) <- solve_multiclass_mva new_vector
          resource_count <- get_resource_count
          result <- foldM (compute_response_time q c) M.empty [0..(resource_count-1)]
          return $ M.insert c result acc

    compute_response_time
      :: M.Map Int Float -> Int -> M.Map Int Float -> Int -> Env (M.Map Int Float)
    compute_response_time q c acc k = 
      get_demand c k >>= \d_ck -> return $ M.insert k (d_ck * (1 + q_k)) acc
      where (Just q_k) = M.lookup k q

    compute_throughput 
      :: M.Map Int (M.Map Int Float) -> M.Map Int Float -> Int -> Env (M.Map Int Float)
    compute_throughput response_times acc c =
      if ns !! c == 0
        then return $ M.insert c 0.0 acc
        else do
          think_time <- get_think_time c
          let (Just class_response_times) = M.lookup c response_times
              total_response_time = sum $ M.elems class_response_times
              throughput = (fromIntegral (ns !! c)) / (think_time + total_response_time)
          return $ M.insert c throughput acc

    compute_q_lengths 
      :: M.Map Int (M.Map Int Float) -> M.Map Int Float 
      -> M.Map Int Float -> Int -> Env (M.Map Int Float)
    compute_q_lengths response_times throughputs acc k = do
      class_count <- get_class_count
      let this_resource = sum $ fmap each_class [0..(class_count-1)]
      return $ M.insert k this_resource acc
      where
        each_class c = 
          let (Just throughput) = M.lookup c throughputs
              (Just response_time_c) = M.lookup c response_times
              (Just response_time) = M.lookup k response_time_c
          in (throughput * response_time)

main :: IO ()
main = do
  let ns  = [3, 1]
      demand = [ [0.105, 0.180, 0.000]
               , [0.375, 0.480, 0.240]
               ]
      think_time = [0.0, 0.0]
      k = 3
      c = 2
      init_state = Mva_Parameters demand k c think_time
      rt = solve_multiclass_mva ns
      result = runIdentity $ runReaderT rt init_state
  putStrLn $ ppShow result

















