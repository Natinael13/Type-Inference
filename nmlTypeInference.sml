(* ml.sml S411c *)


(*****************************************************************)
(*                                                               *)
(*   EXCEPTIONS USED IN LANGUAGES WITH TYPE INFERENCE            *)
(*                                                               *)
(*****************************************************************)

(* exceptions used in languages with type inference S237c *)
exception TypeError of string
exception BugInTypeInference of string


(*****************************************************************)
(*                                                               *)
(*   \FOOTNOTESIZE SHARED: NAMES, ENVIRONMENTS, STRINGS, ERRORS, PRINTING, INTERACTION, STREAMS, \&\ INITIALIZATION *)
(*                                                               *)
(*****************************************************************)

(* \footnotesize shared: names, environments, strings, errors, printing, interaction, streams, \&\ initialization S237a *)
(* for working with curried functions: [[id]], [[fst]], [[snd]], [[pair]], [[curry]], and [[curry3]] S263d *)
fun id x = x
fun fst (x, y) = x
fun snd (x, y) = y
fun pair x y = (x, y)
fun curry  f x y   = f (x, y)
fun curry3 f x y z = f (x, y, z)
(* type declarations for consistency checking *)
val _ = op fst    : ('a * 'b) -> 'a
val _ = op snd    : ('a * 'b) -> 'b
val _ = op pair   : 'a -> 'b -> 'a * 'b
val _ = op curry  : ('a * 'b -> 'c) -> ('a -> 'b -> 'c)
val _ = op curry3 : ('a * 'b * 'c -> 'd) -> ('a -> 'b -> 'c -> 'd)
(* support for names and environments 310a *)
type name = string
(* support for names and environments 310b *)
type 'a env = (name * 'a) list
(* support for names and environments 311a *)
val emptyEnv = []
(* support for names and environments 311b *)
exception NotFound of name
fun find (name, []) = raise NotFound name
  | find (name, (x, v)::tail) = if name = x then v else find (name, tail)
(* support for names and environments 312a *)
fun isbound (name, []) = false
  | isbound (name, (x, v)::tail) = name = x orelse isbound (name, tail)
(* support for names and environments 312b *)
fun bind (name, v, rho) =
  (name, v) :: rho
(* support for names and environments 312c *)
exception BindListLength
fun bindList (x::vars, v::vals, rho) = bindList (vars, vals, bind (x, v, rho))
  | bindList ([], [], rho) = rho
  | bindList _ = raise BindListLength

fun mkEnv (xs, vs) = bindList (xs, vs, emptyEnv)
(* support for names and environments 312d *)
(* composition *)
infix 6 <+>
fun pairs <+> pairs' = pairs' @ pairs
(* type declarations for consistency checking *)
val _ = op emptyEnv : 'a env
(* type declarations for consistency checking *)
val _ = op find : name * 'a env -> 'a
(* type declarations for consistency checking *)
val _ = op bind : name * 'a * 'a env -> 'a env
(* type declarations for consistency checking *)
val _ = op bindList : name list * 'a list * 'a env -> 'a env
val _ = op mkEnv    : name list * 'a list -> 'a env
(* type declarations for consistency checking *)
val _ = op <+> : 'a env * 'a env -> 'a env
(* support for names and environments S366d *)
fun duplicatename [] = NONE
  | duplicatename (x::xs) =
      if List.exists (fn x' => x' = x) xs then
        SOME x
      else
        duplicatename xs
(* type declarations for consistency checking *)
val _ = op duplicatename : name list -> name option
(* support for detecting and signaling errors detected at run time S366c *)
exception RuntimeError of string (* error message *)
(* support for detecting and signaling errors detected at run time S366e *)
fun errorIfDups (what, xs, context) =
  case duplicatename xs
    of NONE   => ()
     | SOME x => raise RuntimeError (what ^ " " ^ x ^ " appears twice in " ^
                                                                        context)
(* type declarations for consistency checking *)
val _ = op errorIfDups : string * name list * string -> unit
(* support for detecting and signaling errors detected at run time S366f *)
exception InternalError of string (* bug in the interpreter *)
(* list functions not provided by \sml's initial basis S241b *)
fun zip3 ([], [], []) = []
  | zip3 (x::xs, y::ys, z::zs) = (x, y, z) :: zip3 (xs, ys, zs)
  | zip3 _ = raise ListPair.UnequalLengths

fun unzip3 [] = ([], [], [])
  | unzip3 (trip::trips) =
      let val (x,  y,  z)  = trip
          val (xs, ys, zs) = unzip3 trips
      in  (x::xs, y::ys, z::zs)
      end
(* list functions not provided by \sml's initial basis S241c *)
val reverse = rev
(* list functions not provided by \sml's initial basis S242a *)
fun optionList [] = SOME []
  | optionList (NONE :: _) = NONE
  | optionList (SOME x :: rest) =
      (case optionList rest
         of SOME xs => SOME (x :: xs)
          | NONE    => NONE)
(* utility functions for string manipulation and printing S238a *)
fun println  s = (print s; print "\n")
fun eprint   s = TextIO.output (TextIO.stdErr, s)
fun eprintln s = (eprint s; eprint "\n")
(* utility functions for string manipulation and printing S238b *)
val xprinter = ref print
fun xprint   s = !xprinter s
fun xprintln s = (xprint s; xprint "\n")
(* utility functions for string manipulation and printing S238c *)
fun tryFinally f x post =
  (f x handle e => (post (); raise e)) before post ()

fun withXprinter xp f x =
  let val oxp = !xprinter
      val ()  = xprinter := xp
  in  tryFinally f x (fn () => xprinter := oxp)
  end
(* utility functions for string manipulation and printing S238d *)
fun bprinter () =
  let val buffer = ref []
      fun bprint s = buffer := s :: !buffer
      fun contents () = concat (rev (!buffer))
  in  (bprint, contents)
  end
(* utility functions for string manipulation and printing S238e *)
fun predefinedFunctionError s = eprintln ("while reading predefined functions, "
                                                                            ^ s)
(* utility functions for string manipulation and printing S238f *)
fun intString n =
  String.map (fn #"~" => #"-" | c => c) (Int.toString n)
(* utility functions for string manipulation and printing S238g *)
fun plural what [x] = what
  | plural what _   = what ^ "s"

fun countString xs what =
  intString (length xs) ^ " " ^ plural what xs
(* utility functions for string manipulation and printing S239a *)
fun separate (zero, sep) = 
  (* list with separator *)
  let fun s []     = zero
        | s [x]    = x
        | s (h::t) = h ^ sep ^ s t
  in  s
end
val spaceSep = separate ("", " ")   (* list separated by spaces *)
val commaSep = separate ("", ", ")  (* list separated by commas *)
(* type declarations for consistency checking *)
val _ = op intString : int -> string
(* type declarations for consistency checking *)
val _ = op spaceSep :                    string list -> string
val _ = op commaSep :                    string list -> string
val _ = op separate : string * string -> string list -> string
(* utility functions for string manipulation and printing S239b *)
fun printUTF8 code =
  let val w = Word.fromInt code
      val (&, >>) = (Word.andb, Word.>>)
      infix 6 & >>
      val _ = if (w & 0wx1fffff) <> w then
                raise RuntimeError (intString code ^
                                    " does not represent a Unicode code point")
              else
                 ()
      val printbyte = xprint o str o chr o Word.toInt
      fun prefix byte byte' = Word.orb (byte, byte')
  in  if w > 0wxffff then
        app printbyte [ prefix 0wxf0  (w >> 0w18)
                      , prefix 0wx80 ((w >> 0w12) & 0wx3f)
                      , prefix 0wx80 ((w >>  0w6) & 0wx3f)
                      , prefix 0wx80 (w       & 0wx3f)
                      ]
      else if w > 0wx7ff then
        app printbyte [ prefix 0wxe0  (w >> 0w12)
                      , prefix 0wx80 ((w >>  0w6) & 0wx3f)
                      , prefix 0wx80 (w         & 0wx3f)
                      ]
      else if w > 0wx7f then
        app printbyte [ prefix 0wxc0  (w >>  0w6)
                      , prefix 0wx80 (w         & 0wx3f)
                      ]
      else
        printbyte w
  end
(* utility functions for string manipulation and printing S239c *)
fun fnvHash s =
  let val offset_basis = 0wx011C9DC5 : Word.word  (* trim the high bit *)
      val fnv_prime    = 0w16777619  : Word.word
      fun update (c, hash) = Word.xorb (hash, Word.fromInt (ord c)) * fnv_prime
      fun int w = Word.toIntX w handle Overflow => Word.toInt (Word.andb (w,
                                                                     0wxffffff))
  in  int (foldl update offset_basis (explode s))
  end
(* type declarations for consistency checking *)
val _ = op fnvHash : string -> int
(* utility functions for string manipulation and printing S240a *)
fun stripNumericSuffix s =
      let fun stripPrefix []         = s   (* don't let things get empty *)
            | stripPrefix (#"-"::[]) = s
            | stripPrefix (#"-"::cs) = implode (reverse cs)
            | stripPrefix (c   ::cs) = if Char.isDigit c then stripPrefix cs
                                       else implode (reverse (c::cs))
      in  stripPrefix (reverse (explode s))
      end
(* support for representing errors as \ml\ values S243b *)
datatype 'a error = OK of 'a | ERROR of string
(* support for representing errors as \ml\ values S244a *)
infix 1 >>=
fun (OK x)      >>= k  =  k x
  | (ERROR msg) >>= k  =  ERROR msg
(* type declarations for consistency checking *)
val _ = op zip3   : 'a list * 'b list * 'c list -> ('a * 'b * 'c) list
val _ = op unzip3 : ('a * 'b * 'c) list -> 'a list * 'b list * 'c list
(* type declarations for consistency checking *)
val _ = op optionList : 'a option list -> 'a list option
(* type declarations for consistency checking *)
val _ = op >>= : 'a error * ('a -> 'b error) -> 'b error
(* support for representing errors as \ml\ values S244b *)
infix 1 >>=+
fun e >>=+ k'  =  e >>= (OK o k')
(* type declarations for consistency checking *)
val _ = op >>=+ : 'a error * ('a -> 'b) -> 'b error
(* support for representing errors as \ml\ values S244c *)
fun errorList es =
  let fun cons (OK x, OK xs) = OK (x :: xs)
        | cons (ERROR m1, ERROR m2) = ERROR (m1 ^ "; " ^ m2)
        | cons (ERROR m, OK _) = ERROR m
        | cons (OK _, ERROR m) = ERROR m
  in  foldr cons (OK []) es
  end
(* type declarations for consistency checking *)
val _ = op errorList : 'a error list -> 'a list error
(* support for representing errors as \ml\ values S245a *)
fun errorLabel s (OK x) = OK x
  | errorLabel s (ERROR msg) = ERROR (s ^ msg)
(* type [[interactivity]] plus related functions and value S368a *)
datatype input_interactivity = PROMPTING | NOT_PROMPTING
(* type [[interactivity]] plus related functions and value S368b *)
datatype output_interactivity = PRINTING | NOT_PRINTING
(* type [[interactivity]] plus related functions and value S368c *)
type interactivity = 
  input_interactivity * output_interactivity
val noninteractive = 
  (NOT_PROMPTING, NOT_PRINTING)
fun prompts (PROMPTING,     _) = true
  | prompts (NOT_PROMPTING, _) = false
fun prints (_, PRINTING)     = true
  | prints (_, NOT_PRINTING) = false
(* type declarations for consistency checking *)
type interactivity = interactivity
val _ = op noninteractive : interactivity
val _ = op prompts : interactivity -> bool
val _ = op prints  : interactivity -> bool
(* simple implementations of set operations S240b *)
type 'a set = 'a list
val emptyset = []
fun member x = 
  List.exists (fn y => y = x)
fun insert (x, ys) = 
  if member x ys then ys else x::ys
fun union (xs, ys) = foldl insert ys xs
fun inter (xs, ys) =
  List.filter (fn x => member x ys) xs
fun diff  (xs, ys) = 
  List.filter (fn x => not (member x ys)) xs
(* type declarations for consistency checking *)
type 'a set = 'a set
val _ = op emptyset : 'a set
val _ = op member   : ''a -> ''a set -> bool
val _ = op insert   : ''a     * ''a set  -> ''a set
val _ = op union    : ''a set * ''a set  -> ''a set
val _ = op inter    : ''a set * ''a set  -> ''a set
val _ = op diff     : ''a set * ''a set  -> ''a set
(* collections with mapping and combining functions S240c *)
datatype 'a collection = C of 'a set
fun elemsC (C xs) = xs
fun singleC x     = C [x]
val emptyC        = C []
(* type declarations for consistency checking *)
type 'a collection = 'a collection
val _ = op elemsC  : 'a collection -> 'a set
val _ = op singleC : 'a -> 'a collection
val _ = op emptyC  :       'a collection
(* collections with mapping and combining functions S241a *)
fun joinC     (C xs) = C (List.concat (map elemsC xs))
fun mapC  f   (C xs) = C (map f xs)
fun filterC p (C xs) = C (List.filter p xs)
fun mapC2 f (xc, yc) = joinC (mapC (fn x => mapC (fn y => f (x, y)) yc) xc)
(* type declarations for consistency checking *)
val _ = op joinC   : 'a collection collection -> 'a collection
val _ = op mapC    : ('a -> 'b)      -> ('a collection -> 'b collection)
val _ = op filterC : ('a -> bool)    -> ('a collection -> 'a collection)
val _ = op mapC2   : ('a * 'b -> 'c) -> ('a collection * 'b collection -> 'c
                                                                     collection)
(* suspensions S249a *)
datatype 'a action
  = PENDING  of unit -> 'a
  | PRODUCED of 'a

type 'a susp = 'a action ref
(* type declarations for consistency checking *)
type 'a susp = 'a susp
(* suspensions S249b *)
fun delay f = ref (PENDING f)
fun demand cell =
  case !cell
    of PENDING f =>  let val result = f ()
                     in  (cell := PRODUCED result; result)
                     end
     | PRODUCED v => v
(* type declarations for consistency checking *)
val _ = op delay  : (unit -> 'a) -> 'a susp
val _ = op demand : 'a susp -> 'a
(* streams S250a *)
datatype 'a stream 
  = EOS
  | :::       of 'a * 'a stream
  | SUSPENDED of 'a stream susp
infixr 3 :::
(* streams S250b *)
fun streamGet EOS = NONE
  | streamGet (x ::: xs)    = SOME (x, xs)
  | streamGet (SUSPENDED s) = streamGet (demand s)
(* streams S250c *)
fun streamOfList xs = 
  foldr op ::: EOS xs
(* type declarations for consistency checking *)
val _ = op streamGet : 'a stream -> ('a * 'a stream) option
(* type declarations for consistency checking *)
val _ = op streamOfList : 'a list -> 'a stream
(* streams S250d *)
fun listOfStream xs =
  case streamGet xs
    of NONE => []
     | SOME (x, xs) => x :: listOfStream xs
(* streams S251a *)
fun delayedStream action = 
  SUSPENDED (delay action)
(* type declarations for consistency checking *)
val _ = op listOfStream : 'a stream -> 'a list
(* type declarations for consistency checking *)
val _ = op delayedStream : (unit -> 'a stream) -> 'a stream
(* streams S251b *)
fun streamOfEffects action =
  delayedStream (fn () => case action () of NONE   => EOS
                                          | SOME a => a ::: streamOfEffects
                                                                         action)
(* type declarations for consistency checking *)
val _ = op streamOfEffects : (unit -> 'a option) -> 'a stream
(* streams S251c *)
type line = string
fun filelines infile = 
  streamOfEffects (fn () => TextIO.inputLine infile)
(* type declarations for consistency checking *)
type line = line
val _ = op filelines : TextIO.instream -> line stream
(* streams S251d *)
fun streamRepeat x =
  delayedStream (fn () => x ::: streamRepeat x)
(* type declarations for consistency checking *)
val _ = op streamRepeat : 'a -> 'a stream
(* streams S251e *)
fun streamOfUnfold next state =
  delayedStream (fn () => case next state
                            of NONE => EOS
                             | SOME (a, state') => a ::: streamOfUnfold next
                                                                         state')
(* type declarations for consistency checking *)
val _ = op streamOfUnfold : ('b -> ('a * 'b) option) -> 'b -> 'a stream
(* streams S252a *)
val naturals = 
  streamOfUnfold (fn n => SOME (n, n+1)) 0   (* 0 to infinity *)
(* type declarations for consistency checking *)
val _ = op naturals : int stream
(* streams S252b *)
fun preStream (pre, xs) = 
  streamOfUnfold (fn xs => (pre (); streamGet xs)) xs
(* streams S252c *)
fun postStream (xs, post) =
  streamOfUnfold (fn xs => case streamGet xs
                             of NONE => NONE
                              | head as SOME (x, _) => (post x; head)) xs
(* type declarations for consistency checking *)
val _ = op preStream : (unit -> unit) * 'a stream -> 'a stream
(* type declarations for consistency checking *)
val _ = op postStream : 'a stream * ('a -> unit) -> 'a stream
(* streams S252d *)
fun streamMap f xs =
  delayedStream (fn () => case streamGet xs
                            of NONE => EOS
                             | SOME (x, xs) => f x ::: streamMap f xs)
(* type declarations for consistency checking *)
val _ = op streamMap : ('a -> 'b) -> 'a stream -> 'b stream
(* streams S253a *)
fun streamFilter p xs =
  delayedStream (fn () => case streamGet xs
                            of NONE => EOS
                             | SOME (x, xs) => if p x then x ::: streamFilter p
                                                                              xs
                                               else streamFilter p xs)
(* type declarations for consistency checking *)
val _ = op streamFilter : ('a -> bool) -> 'a stream -> 'a stream
(* streams S253b *)
fun streamFold f z xs =
  case streamGet xs of NONE => z
                     | SOME (x, xs) => streamFold f (f (x, z)) xs
(* type declarations for consistency checking *)
val _ = op streamFold : ('a * 'b -> 'b) -> 'b -> 'a stream -> 'b
(* streams S253c *)
fun streamZip (xs, ys) =
  delayedStream
  (fn () => case (streamGet xs, streamGet ys)
              of (SOME (x, xs), SOME (y, ys)) => (x, y) ::: streamZip (xs, ys)
               | _ => EOS)
(* streams S253d *)
fun streamConcat xss =
  let fun get (xs, xss) =
        case streamGet xs
          of SOME (x, xs) => SOME (x, (xs, xss))
           | NONE => case streamGet xss
                       of SOME (xs, xss) => get (xs, xss)
                        | NONE => NONE
  in  streamOfUnfold get (EOS, xss)
  end
(* type declarations for consistency checking *)
val _ = op streamZip : 'a stream * 'b stream -> ('a * 'b) stream
(* type declarations for consistency checking *)
val _ = op streamConcat : 'a stream stream -> 'a stream
(* streams S253e *)
fun streamConcatMap f xs = streamConcat (streamMap f xs)
(* type declarations for consistency checking *)
val _ = op streamConcatMap : ('a -> 'b stream) -> 'a stream -> 'b stream
(* streams S253f *)
infix 5 @@@
fun xs @@@ xs' = streamConcat (streamOfList [xs, xs'])
(* type declarations for consistency checking *)
val _ = op @@@ : 'a stream * 'a stream -> 'a stream
(* streams S254a *)
fun streamTake (0, xs) = []
  | streamTake (n, xs) =
      case streamGet xs
        of SOME (x, xs) => x :: streamTake (n-1, xs)
         | NONE => []
(* type declarations for consistency checking *)
val _ = op streamTake : int * 'a stream -> 'a list
(* streams S254b *)
fun streamDrop (0, xs) = xs
  | streamDrop (n, xs) =
      case streamGet xs
        of SOME (_, xs) => streamDrop (n-1, xs)
         | NONE => EOS
(* type declarations for consistency checking *)
val _ = op streamDrop : int * 'a stream -> 'a stream
(* stream transformers and their combinators S261a *)
type ('a, 'b) xformer = 
  'a stream -> ('b error * 'a stream) option
(* type declarations for consistency checking *)
type ('a, 'b) xformer = ('a, 'b) xformer
(* stream transformers and their combinators S261b *)
fun pure y = fn xs => SOME (OK y, xs)
(* type declarations for consistency checking *)
val _ = op pure : 'b -> ('a, 'b) xformer
(* stream transformers and their combinators S263a *)
infix 3 <*>
fun tx_f <*> tx_b =
  fn xs => case tx_f xs
             of NONE => NONE
              | SOME (ERROR msg, xs) => SOME (ERROR msg, xs)
              | SOME (OK f, xs) =>
                  case tx_b xs
                    of NONE => NONE
                     | SOME (ERROR msg, xs) => SOME (ERROR msg, xs)
                     | SOME (OK y, xs) => SOME (OK (f y), xs)
(* type declarations for consistency checking *)
val _ = op <*> : ('a, 'b -> 'c) xformer * ('a, 'b) xformer -> ('a, 'c) xformer
(* stream transformers and their combinators S263b *)
infixr 4 <$>
fun f <$> p = pure f <*> p
(* type declarations for consistency checking *)
val _ = op <$> : ('b -> 'c) * ('a, 'b) xformer -> ('a, 'c) xformer
(* stream transformers and their combinators S263c *)
infixr 3 <~>
fun f <~> a = curry fst <$> f <*> a
(* stream transformers and their combinators S264a *)
infix 1 <|>
fun t1 <|> t2 = (fn xs => case t1 xs of SOME y => SOME y | NONE => t2 xs) 
(* type declarations for consistency checking *)
val _ = op <|> : ('a, 'b) xformer * ('a, 'b) xformer -> ('a, 'b) xformer
(* stream transformers and their combinators S264b *)
fun pzero _ = NONE
(* stream transformers and their combinators S264c *)
fun anyParser ts = 
  foldr op <|> pzero ts
(* type declarations for consistency checking *)
val _ = op pzero : ('a, 'b) xformer
(* type declarations for consistency checking *)
val _ = op anyParser : ('a, 'b) xformer list -> ('a, 'b) xformer
(* stream transformers and their combinators S264d *)
infix 6 <* *>
fun p1 <*  p2 = curry fst <$> p1 <*> p2
fun p1  *> p2 = curry snd <$> p1 <*> p2

infixr 4 <$
fun v <$ p = (fn _ => v) <$> p
(* type declarations for consistency checking *)
val _ = op <*  : ('a, 'b) xformer * ('a, 'c) xformer -> ('a, 'b) xformer
val _ = op  *> : ('a, 'b) xformer * ('a, 'c) xformer -> ('a, 'c) xformer
val _ = op <$  : 'b               * ('a, 'c) xformer -> ('a, 'b) xformer
(* stream transformers and their combinators S265a *)
fun one xs = case streamGet xs
               of NONE => NONE
                | SOME (x, xs) => SOME (OK x, xs)
(* type declarations for consistency checking *)
val _ = op one : ('a, 'a) xformer
(* stream transformers and their combinators S265b *)
fun eos xs = case streamGet xs
               of NONE => SOME (OK (), EOS)
                | SOME _ => NONE
(* type declarations for consistency checking *)
val _ = op eos : ('a, unit) xformer
(* stream transformers and their combinators S265c *)
fun peek tx xs =
  case tx xs of SOME (OK y, _) => SOME y
              | _ => NONE
(* type declarations for consistency checking *)
val _ = op peek : ('a, 'b) xformer -> 'a stream -> 'b option
(* stream transformers and their combinators S265d *)
fun rewind tx xs =
  case tx xs of SOME (ey, _) => SOME (ey, xs)
              | NONE => NONE
(* type declarations for consistency checking *)
val _ = op rewind : ('a, 'b) xformer -> ('a, 'b) xformer
(* stream transformers and their combinators S266a *)
fun sat p tx xs =
  case tx xs
    of answer as SOME (OK y, xs) => if p y then answer else NONE
     | answer => answer
(* type declarations for consistency checking *)
val _ = op sat : ('b -> bool) -> ('a, 'b) xformer -> ('a, 'b) xformer
(* stream transformers and their combinators S266b *)
fun eqx y = 
  sat (fn y' => y = y') 
(* type declarations for consistency checking *)
val _ = op eqx : ''b -> ('a, ''b) xformer -> ('a, ''b) xformer
(* stream transformers and their combinators S266c *)
infixr 4 <$>?
fun f <$>? tx =
  fn xs => case tx xs
             of NONE => NONE
              | SOME (ERROR msg, xs) => SOME (ERROR msg, xs)
              | SOME (OK y, xs) =>
                  case f y
                    of NONE => NONE
                     | SOME z => SOME (OK z, xs)
(* type declarations for consistency checking *)
val _ = op <$>? : ('b -> 'c option) * ('a, 'b) xformer -> ('a, 'c) xformer
(* stream transformers and their combinators S266d *)
infix 3 <&>
fun t1 <&> t2 = fn xs =>
  case t1 xs
    of SOME (OK _, _) => t2 xs
     | SOME (ERROR _, _) => NONE    
     | NONE => NONE
(* type declarations for consistency checking *)
val _ = op <&> : ('a, 'b) xformer * ('a, 'c) xformer -> ('a, 'c) xformer
(* stream transformers and their combinators S267a *)
fun notFollowedBy t xs =
  case t xs
    of NONE => SOME (OK (), xs)
     | SOME _ => NONE
(* type declarations for consistency checking *)
val _ = op notFollowedBy : ('a, 'b) xformer -> ('a, unit) xformer
(* stream transformers and their combinators S267b *)
fun many t = 
  curry (op ::) <$> t <*> (fn xs => many t xs) <|> pure []
(* type declarations for consistency checking *)
val _ = op many  : ('a, 'b) xformer -> ('a, 'b list) xformer
(* stream transformers and their combinators S267c *)
fun many1 t = 
  curry (op ::) <$> t <*> many t
(* type declarations for consistency checking *)
val _ = op many1 : ('a, 'b) xformer -> ('a, 'b list) xformer
(* stream transformers and their combinators S267d *)
fun optional t = 
  SOME <$> t <|> pure NONE
(* type declarations for consistency checking *)
val _ = op optional : ('a, 'b) xformer -> ('a, 'b option) xformer
(* stream transformers and their combinators S268a *)
infix 2 <*>!
fun tx_ef <*>! tx_x =
  fn xs => case (tx_ef <*> tx_x) xs
             of NONE => NONE
              | SOME (OK (OK y),      xs) => SOME (OK y,      xs)
              | SOME (OK (ERROR msg), xs) => SOME (ERROR msg, xs)
              | SOME (ERROR msg,      xs) => SOME (ERROR msg, xs)
infixr 4 <$>!
fun ef <$>! tx_x = pure ef <*>! tx_x
(* type declarations for consistency checking *)
val _ = op <*>! : ('a, 'b -> 'c error) xformer * ('a, 'b) xformer -> ('a, 'c)
                                                                         xformer
val _ = op <$>! : ('b -> 'c error)             * ('a, 'b) xformer -> ('a, 'c)
                                                                         xformer
(* support for source-code locations and located streams S254d *)
type srcloc = string * int
fun srclocString (source, line) =
  source ^ ", line " ^ intString line
(* support for source-code locations and located streams S254e *)
datatype error_format = WITH_LOCATIONS | WITHOUT_LOCATIONS
val toplevel_error_format = ref WITH_LOCATIONS
(* support for source-code locations and located streams S255a *)
fun synerrormsg (source, line) strings =
  if !toplevel_error_format = WITHOUT_LOCATIONS andalso source =
                                                                "standard input"
  then
    concat ("syntax error: " :: strings)
  else    
    concat ("syntax error in " :: srclocString (source, line) :: ": " :: strings
                                                                               )

(* support for source-code locations and located streams S255b *)
exception Located of srcloc * exn
(* support for source-code locations and located streams S255c *)
type 'a located = srcloc * 'a
(* type declarations for consistency checking *)
type srcloc = srcloc
val _ = op srclocString : srcloc -> string
(* type declarations for consistency checking *)
type 'a located = 'a located
(* support for source-code locations and located streams S255d *)
fun atLoc loc f a =
  f a handle e as RuntimeError _ => raise Located (loc, e)
           | e as NotFound _     => raise Located (loc, e)
           (* more handlers for [[atLoc]] S255f *)
           | e as IO.Io _   => raise Located (loc, e)
           | e as Div       => raise Located (loc, e)
           | e as Overflow  => raise Located (loc, e)
           | e as Subscript => raise Located (loc, e)
           | e as Size      => raise Located (loc, e)
           (* more handlers for [[atLoc]] ((type-inference)) S411a *)
           | e as TypeError _          => raise Located (loc, e)
           | e as BugInTypeInference _ => raise Located (loc, e)
(* type declarations for consistency checking *)
val _ = op atLoc : srcloc -> ('a -> 'b) -> ('a -> 'b)
(* support for source-code locations and located streams S255e *)
fun located f (loc, a) = atLoc loc f a
fun leftLocated f ((loc, a), b) = atLoc loc f (a, b)
(* type declarations for consistency checking *)
val _ = op located : ('a -> 'b) -> ('a located -> 'b)
val _ = op leftLocated : ('a * 'b -> 'c) -> ('a located * 'b -> 'c)
(* support for source-code locations and located streams S255g *)
fun fillComplaintTemplate (s, maybeLoc) =
  let val string_to_fill = " <at loc>"
      val (prefix, atloc) = Substring.position string_to_fill (Substring.full s)
      val suffix = Substring.triml (size string_to_fill) atloc
      val splice_in =
        Substring.full (case maybeLoc
                          of NONE => ""
                           | SOME (loc as (file, line)) =>
                               if      !toplevel_error_format =
                                                               WITHOUT_LOCATIONS
                               andalso file = "standard input"
                               then
                                 ""
                               else
                                 " in " ^ srclocString loc)
  in  if Substring.size atloc = 0 then (* <at loc> is not present *)
        s
      else
        Substring.concat [prefix, splice_in, suffix]
  end
fun fillAtLoc (s, loc) = fillComplaintTemplate (s, SOME loc)
fun stripAtLoc s = fillComplaintTemplate (s, NONE)
(* type declarations for consistency checking *)
val _ = op fillComplaintTemplate : string * srcloc option -> string
(* support for source-code locations and located streams S256a *)
fun errorAt msg loc = 
  ERROR (synerrormsg loc [msg])
(* support for source-code locations and located streams S256b *)
fun locatedStream (streamname, inputs) =
  let val locations = streamZip (streamRepeat streamname, streamDrop (1,
                                                                      naturals))
  in  streamZip (locations, inputs)
  end
(* type declarations for consistency checking *)
val _ = op errorAt : string -> srcloc -> 'a error
(* type declarations for consistency checking *)
val _ = op locatedStream : string * line stream -> line located stream
(* streams that track line boundaries S272a *)
datatype 'a eol_marked
  = EOL of int (* number of the line that ends here *)
  | INLINE of 'a

fun drainLine EOS = EOS
  | drainLine (SUSPENDED s)     = drainLine (demand s)
  | drainLine (EOL _    ::: xs) = xs
  | drainLine (INLINE _ ::: xs) = drainLine xs
(* streams that track line boundaries S272b *)
local 
  fun asEol (EOL n) = SOME n
    | asEol (INLINE _) = NONE
  fun asInline (INLINE x) = SOME x
    | asInline (EOL _)    = NONE
in
  fun eol    xs = (asEol    <$>? one) xs
  fun inline xs = (asInline <$>? many eol *> one) xs
  fun srcloc xs = rewind (fst <$> inline) xs
end
(* type declarations for consistency checking *)
type 'a eol_marked = 'a eol_marked
val _ = op drainLine : 'a eol_marked stream -> 'a eol_marked stream
(* type declarations for consistency checking *)
val _ = op eol      : ('a eol_marked, int) xformer
val _ = op inline   : ('a eol_marked, 'a)  xformer
val _ = op srcloc   : ('a located eol_marked, srcloc) xformer
(* support for lexical analysis S268b *)
type 'a lexer = (char, 'a) xformer
(* type declarations for consistency checking *)
type 'a lexer = 'a lexer
(* support for lexical analysis S268c *)
fun isDelim c =
  Char.isSpace c orelse Char.contains "()[]{};" c
(* type declarations for consistency checking *)
val _ = op isDelim : char -> bool
(* support for lexical analysis S270a *)
val whitespace = many (sat Char.isSpace one)
(* type declarations for consistency checking *)
val _ = op whitespace : char list lexer
(* support for lexical analysis S270b *)
fun intChars isDelim = 
  (curry (op ::) <$> eqx #"-" one <|> pure id) <*> many1 (sat Char.isDigit one)
                                                                              <*
  notFollowedBy (sat (not o isDelim) one)
(* type declarations for consistency checking *)
val _ = op intChars : (char -> bool) -> char list lexer
(* support for lexical analysis S270c *)
fun intFromChars (#"-" :: cs) = 
      intFromChars cs >>=+ Int.~
  | intFromChars cs =
      (OK o valOf o Int.fromString o implode) cs
      handle Overflow => ERROR
                        "this interpreter can't read arbitrarily large integers"
(* type declarations for consistency checking *)
val _ = op intFromChars : char list -> int error
(* support for lexical analysis S270d *)
fun intToken isDelim =
  intFromChars <$>! intChars isDelim
(* type declarations for consistency checking *)
val _ = op intToken : (char -> bool) -> int lexer
(* support for lexical analysis S271a *)
datatype bracket_shape = ROUND | SQUARE | CURLY

fun leftString ROUND  = "("
  | leftString SQUARE = "["
  | leftString CURLY  = "{"
fun rightString ROUND  = ")"
  | rightString SQUARE = "]"
  | rightString CURLY = "}"
(* support for lexical analysis S271b *)
datatype 'a plus_brackets
  = LEFT  of bracket_shape
  | RIGHT of bracket_shape
  | PRETOKEN of 'a

fun bracketLexer pretoken
  =  LEFT  ROUND  <$ eqx #"(" one
 <|> LEFT  SQUARE <$ eqx #"[" one
 <|> LEFT  CURLY  <$ eqx #"{" one
 <|> RIGHT ROUND  <$ eqx #")" one
 <|> RIGHT SQUARE <$ eqx #"]" one
 <|> RIGHT CURLY  <$ eqx #"}" one
 <|> PRETOKEN <$> pretoken

fun plusBracketsString _   (LEFT shape)  = leftString shape
  | plusBracketsString _   (RIGHT shape) = rightString shape
  | plusBracketsString pts (PRETOKEN pt)  = pts pt
(* type declarations for consistency checking *)
type 'a plus_brackets = 'a plus_brackets
val _ = op bracketLexer : 'a lexer -> 'a plus_brackets lexer
(* common parsing code S260 *)
(* combinators and utilities for parsing located streams S272c *)
type ('t, 'a) polyparser = ('t located eol_marked, 'a) xformer
(* combinators and utilities for parsing located streams S273a *)
fun token    stream = (snd <$> inline)      stream
fun noTokens stream = (notFollowedBy token) stream
(* type declarations for consistency checking *)
val _ = op token    : ('t, 't)   polyparser
val _ = op noTokens : ('t, unit) polyparser
(* combinators and utilities for parsing located streams S273b *)
fun @@ p = pair <$> srcloc <*> p
(* type declarations for consistency checking *)
val _ = op @@ : ('t, 'a) polyparser -> ('t, 'a located) polyparser
(* combinators and utilities for parsing located streams S273c *)
infix 0 <?>
fun p <?> what = p <|> errorAt ("expected " ^ what) <$>! srcloc
(* type declarations for consistency checking *)
val _ = op <?> : ('t, 'a) polyparser * string -> ('t, 'a) polyparser
(* combinators and utilities for parsing located streams S273d *)
infix 4 <!>
fun p <!> msg =
  fn tokens => (case p tokens
                  of SOME (OK _, unread) =>
                       (case peek srcloc tokens
                          of SOME loc => SOME (errorAt msg loc, unread)
                           | NONE => NONE)
                   | _ => NONE)
(* type declarations for consistency checking *)
val _ = op <!> : ('t, 'a) polyparser * string -> ('t, 'b) polyparser
(* combinators and utilities for parsing located streams S277c *)
fun nodups (what, context) (loc, names) =
  let fun dup [] = OK names
        | dup (x::xs) = if List.exists (fn y : string => y = x) xs then
                          errorAt (what ^ " " ^ x ^ " appears twice in " ^
                                                                    context) loc
                        else
                          dup xs
  in  dup names
  end
(* type declarations for consistency checking *)
val _ = op nodups : string * string -> srcloc * name list -> name list error
(* transformers for interchangeable brackets S274 *)
fun notCurly (_, CURLY) = false
  | notCurly _          = true

(* left: takes shape, succeeds or fails
   right: takes shape and
      succeeds with right bracket of correct shape
      errors with right bracket of incorrect shape
      fails with token that is not right bracket *)

fun left  tokens = ((fn (loc, LEFT  s) => SOME (loc, s) | _ => NONE) <$>? inline
                                                                        ) tokens
fun right tokens = ((fn (loc, RIGHT s) => SOME (loc, s) | _ => NONE) <$>? inline
                                                                        ) tokens
fun leftCurly tokens = sat (not o notCurly) left tokens

fun atRight expected = rewind right <?> expected

fun badRight msg =
  (fn (loc, shape) => errorAt (msg ^ " " ^ rightString shape) loc) <$>! right
(* transformers for interchangeable brackets S276a *)
type ('t, 'a) pb_parser = ('t plus_brackets, 'a) polyparser
datatype right_result
  = FOUND_RIGHT      of bracket_shape located
  | SCANNED_TO_RIGHT of srcloc  (* location where scanning started *)
  | NO_RIGHT

fun scanToClose tokens = 
  let val loc = getOpt (peek srcloc tokens, ("end of stream", 9999))
      fun scan lpcount tokens =
        (* lpcount is the number of unmatched left parentheses *)
        case tokens
          of EOL _                  ::: tokens => scan lpcount tokens
           | INLINE (_, LEFT  t)    ::: tokens => scan (lpcount+1) tokens
           | INLINE (_, RIGHT t)    ::: tokens => if lpcount = 0 then
                                                    pure (SCANNED_TO_RIGHT loc)
                                                                          tokens
                                                  else
                                                    scan (lpcount-1) tokens
           | INLINE (_, PRETOKEN _) ::: tokens => scan lpcount tokens
           | EOS         => pure NO_RIGHT tokens
           | SUSPENDED s => scan lpcount (demand s)
  in  scan 0 tokens
  end

fun matchingRight tokens = (FOUND_RIGHT <$> right <|> scanToClose) tokens

fun matchBrackets _ (loc, left) _ NO_RIGHT =
      errorAt ("unmatched " ^ leftString left) loc
  | matchBrackets e (loc, left) _ (SCANNED_TO_RIGHT loc') =
      errorAt ("expected " ^ e) loc
  | matchBrackets _ (loc, left) a (FOUND_RIGHT (loc', right)) =
      if left = right then
        OK a
      else
        errorAt (rightString right ^ " does not match " ^ leftString left ^
                 (if loc <> loc' then " at " ^ srclocString loc else "")) loc'
(* type declarations for consistency checking *)
type right_result = right_result
val _ = op matchingRight : ('t, right_result) pb_parser
val _ = op scanToClose   : ('t, right_result) pb_parser
val _ = op matchBrackets : string -> bracket_shape located -> 'a -> right_result
                                                                     -> 'a error
(* transformers for interchangeable brackets S276b *)
fun liberalBracket (expected, p) =
  matchBrackets expected <$> sat notCurly left <*> p <*>! matchingRight
fun bracketKeyword (keyword, expected, p) =
  liberalBracket (expected, keyword *> (p <?> expected))
fun bracket (expected, p) =
  liberalBracket (expected, p <?> expected)
fun curlyBracket (expected, p) =
  matchBrackets expected <$> leftCurly <*> (p <?> expected) <*>! matchingRight
(* type declarations for consistency checking *)
val _ = op bracketKeyword : ('t, 'keyword) pb_parser * string * ('t, 'a)
                                                 pb_parser -> ('t, 'a) pb_parser
(* transformers for interchangeable brackets S277a *)
fun usageParser keyword =
  let val left = eqx #"(" one <|> eqx #"[" one
      val getkeyword = left *> (implode <$> many1 (sat (not o isDelim) one))
  in  fn (usage, p) =>
        case getkeyword (streamOfList (explode usage))
          of SOME (OK k, _) => bracketKeyword (keyword k, usage, p)
           | _ => let exception BadUsage of string in raise BadUsage usage end
  end
(* type declarations for consistency checking *)
val _ = op usageParser : (string -> ('t, string) pb_parser) -> string * ('t, 'a)
                                                 pb_parser -> ('t, 'a) pb_parser
(* transformers for interchangeable brackets S277b *)
fun pretoken stream = ((fn PRETOKEN t => SOME t | _ => NONE) <$>? token) stream
(* code used to debug parsers S277d *)
fun safeTokens stream =
  let fun tokens (seenEol, seenSuspended) =
            let fun get (EOL _         ::: ts) = if seenSuspended then []
                                                 else tokens (true, false) ts
                  | get (INLINE (_, t) ::: ts) = t :: get ts
                  | get  EOS                   = []
                  | get (SUSPENDED (ref (PRODUCED ts))) = get ts
                  | get (SUSPENDED s) = if seenEol then []
                                        else tokens (false, true) (demand s)
            in   get
            end
  in  tokens (false, false) stream
  end
(* type declarations for consistency checking *)
val _ = op safeTokens : 'a located eol_marked stream -> 'a list
(* code used to debug parsers S278a *)
fun showErrorInput asString p tokens =
  case p tokens
    of result as SOME (ERROR msg, rest) =>
         if String.isSubstring " [input: " msg then
           result
         else
           SOME (ERROR (msg ^ " [input: " ^
                        spaceSep (map asString (safeTokens tokens)) ^ "]"),
               rest)
     | result => result
(* type declarations for consistency checking *)
val _ = op showErrorInput : ('t -> string) -> ('t, 'a) polyparser -> ('t, 'a)
                                                                      polyparser
(* code used to debug parsers S278b *)
fun wrapAround tokenString what p tokens =
  let fun t tok = " " ^ tokenString tok
      val _ = app eprint ["Looking for ", what, " at"]
      val _ = app (eprint o t) (safeTokens tokens)
      val _ = eprint "\n"
      val answer = p tokens
      val _ = app eprint [case answer of NONE => "Didn't find " | SOME _ =>
                                                                       "Found ",
                         what, "\n"]
  in  answer
  end handle e => ( app eprint ["Search for ", what, " raised ", exnName e, "\n"
                                                                               ]
                  ; raise e)
(* type declarations for consistency checking *)
val _ = op wrapAround : ('t -> string) -> string -> ('t, 'a) polyparser -> ('t,
                                                                  'a) polyparser
(* streams that issue two forms of prompts S279a *)
fun echoTagStream lines = 
  let fun echoIfTagged line =
        if (String.substring (line, 0, 2) = ";#" handle _ => false) then
          print line
        else
          ()
  in  postStream (lines, echoIfTagged)
  end
(* type declarations for consistency checking *)
val _ = op echoTagStream : line stream -> line stream 
(* streams that issue two forms of prompts S279b *)
fun stripAndReportErrors xs =
  let fun next xs =
        case streamGet xs
          of SOME (ERROR msg, xs) => (eprintln msg; next xs)
           | SOME (OK x, xs) => SOME (x, xs)
           | NONE => NONE
  in  streamOfUnfold next xs
  end
(* type declarations for consistency checking *)
val _ = op stripAndReportErrors : 'a error stream -> 'a stream
(* streams that issue two forms of prompts S279c *)
fun lexLineWith lexer =
  stripAndReportErrors o streamOfUnfold lexer o streamOfList o explode
(* type declarations for consistency checking *)
val _ = op lexLineWith : 't lexer -> line -> 't stream
(* streams that issue two forms of prompts S279d *)
fun parseWithErrors parser =
  let fun adjust (SOME (ERROR msg, tokens)) = SOME (ERROR msg, drainLine tokens)
        | adjust other = other
  in  streamOfUnfold (adjust o parser)
  end
(* type declarations for consistency checking *)
val _ = op parseWithErrors : ('t, 'a) polyparser -> 't located eol_marked stream
                                                              -> 'a error stream
(* streams that issue two forms of prompts S280a *)
type prompts   = { ps1 : string, ps2 : string }
val stdPrompts = { ps1 = "-> ", ps2 = "   " }
val noPrompts  = { ps1 = "", ps2 = "" }
(* type declarations for consistency checking *)
type prompts = prompts
val _ = op stdPrompts : prompts
val _ = op noPrompts  : prompts
(* streams that issue two forms of prompts S280b *)
fun ('t, 'a) interactiveParsedStream (lexer, parser) (name, lines, prompts) =
  let val { ps1, ps2 } = prompts
      val thePrompt = ref ps1
      fun setPrompt ps = fn _ => thePrompt := ps

      val lines = preStream (fn () => print (!thePrompt), echoTagStream lines)

      fun lexAndDecorate (loc, line) =
        let val tokens = postStream (lexLineWith lexer line, setPrompt ps2)
        in  streamMap INLINE (streamZip (streamRepeat loc, tokens)) @@@
            streamOfList [EOL (snd loc)]
        end

      val xdefs_with_errors : 'a error stream = 
        (parseWithErrors parser o streamConcatMap lexAndDecorate o locatedStream
                                                                               )
        (name, lines)
(* type declarations for consistency checking *)
val _ = op interactiveParsedStream : 't lexer * ('t, 'a) polyparser -> string *
                                              line stream * prompts -> 'a stream
val _ = op lexAndDecorate : srcloc * line -> 't located eol_marked stream
  in  
      stripAndReportErrors (preStream (setPrompt ps1, xdefs_with_errors))
  end 
(* common parsing code ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
fun ('t, 'a) finiteStreamOfLine fail (lexer, parser) line =
  let val lines = streamOfList [line] @@@ streamOfEffects fail
      fun lexAndDecorate (loc, line) =
        let val tokens = lexLineWith lexer line
        in  streamMap INLINE (streamZip (streamRepeat loc, tokens)) @@@
            streamOfList [EOL (snd loc)]
        end

      val things_with_errors : 'a error stream = 
        (parseWithErrors parser o streamConcatMap lexAndDecorate o locatedStream
                                                                               )
        ("command line", lines)
  in  
      stripAndReportErrors things_with_errors
  end 
val _ = finiteStreamOfLine :
          (unit -> string option) -> 't lexer * ('t, 'a) polyparser -> line ->
                                                                       'a stream
(* shared utility functions for initializing interpreters S372b *)
fun override_if_testing () =                           (*OMIT*)
  if isSome (OS.Process.getEnv "NOERRORLOC") then      (*OMIT*)
    toplevel_error_format := WITHOUT_LOCATIONS         (*OMIT*)
  else                                                 (*OMIT*)
    ()                                                 (*OMIT*)
fun setup_error_format interactivity =
  if prompts interactivity then
    toplevel_error_format := WITHOUT_LOCATIONS
    before override_if_testing () (*OMIT*)
  else
    toplevel_error_format := WITH_LOCATIONS
    before override_if_testing () (*OMIT*)
(* function application with overflow checking S242b *)
local
  val throttleCPU = case OS.Process.getEnv "BPCOPTIONS"
                      of SOME "nothrottle" => false
                       | _ => true
  val defaultRecursionLimit = 6000 (* about 1/5 of 32,000? *)
  val recursionLimit = ref defaultRecursionLimit
  val evalFuel       = ref 1000000
  datatype checkpoint = RECURSION_LIMIT of int
in
  val defaultEvalFuel = ref (!evalFuel)
  fun withFuel n f x = 
    let val old = !evalFuel
        val _ = evalFuel := n
    in  (f x before evalFuel := old) handle e => (evalFuel := old; raise e)
    end

  fun fuelRemaining () = !evalFuel

  fun checkpointLimit () = RECURSION_LIMIT (!recursionLimit)
  fun restoreLimit (RECURSION_LIMIT n) = recursionLimit := n

  fun applyCheckingOverflow f =
    if !recursionLimit <= 0 then
      raise RuntimeError "recursion too deep"
    else if throttleCPU andalso !evalFuel <= 0 then
      (evalFuel := !defaultEvalFuel; raise RuntimeError "CPU time exhausted")
    else
      let val _ = recursionLimit := !recursionLimit - 1
          val _ = evalFuel        := !evalFuel - 1
      in  fn arg => f arg before (recursionLimit := !recursionLimit + 1)
      end
  fun resetOverflowCheck () = ( recursionLimit := defaultRecursionLimit
                              ; evalFuel := !defaultEvalFuel
                              )
end
(* function [[forward]], for mutual recursion through mutable reference cells S243a *)
fun forward what _ =
  let exception UnresolvedForwardDeclaration of string
  in  raise UnresolvedForwardDeclaration what
  end
exception LeftAsExercise of string



(*****************************************************************)
(*                                                               *)
(*   HINDLEY-MILNER TYPES WITH NAMED TYPE CONSTRUCTORS           *)
(*                                                               *)
(*****************************************************************)

(* Hindley-Milner types with named type constructors S408c *)
(* definitions of [[tycon]], [[eqTycon]], and [[tyconString]] for named type constructors 419a *)
type tycon = name
fun eqTycon (mu, mu') = mu = mu'
fun tyconString mu = mu
(* type declarations for consistency checking *)
type tycon = tycon
val _ = op eqTycon : tycon * tycon -> bool
val _ = op tyconString : tycon -> string
(* representation of Hindley-Milner types 418 *)
type tyvar  = name
datatype ty = TYVAR  of tyvar               (* type variable alpha *)
            | TYCON  of tycon               (* type constructor mu *)
            | CONAPP of ty * ty list        (* type-level application *)

datatype type_scheme = FORALL of tyvar list * ty
(* sets of free type variables in Hindley-Milner types 442 *)
fun freetyvars t =
  let fun f (TYVAR v,          ftvs) = insert (v, ftvs)
        | f (TYCON _,          ftvs) = ftvs
        | f (CONAPP (ty, tys), ftvs) = foldl f (f (ty, ftvs)) tys
  in  reverse (f (t, emptyset))
  end  
(* type declarations for consistency checking *)
val _ = op freetyvars : ty -> name set
val funtycon = "function"
(* functions that create or compare Hindley-Milner types with named type constructors 422c *)
val inttype  = TYCON "int"
val booltype = TYCON "bool"
val symtype  = TYCON "sym"
val alpha    = TYVAR "a"
val beta     = TYVAR "b"
val unittype = TYCON "unit"
fun listtype ty = 
  CONAPP (TYCON "list", [ty])
fun pairtype (x, y) =
  CONAPP (TYCON "pair", [x, y])
fun funtype (args, result) = 
  CONAPP (TYCON "function", [CONAPP (TYCON "arguments", args), result])
fun asFuntype (CONAPP (TYCON "function",
                       [CONAPP (TYCON "arguments", args), result])) =
      SOME (args, result)
  | asFuntype _ = NONE
(* type declarations for consistency checking *)
val _ = op inttype   : ty
val _ = op booltype  : ty
val _ = op symtype   : ty
val _ = op alpha     : ty
val _ = op beta      : ty
val _ = op unittype  : ty
val _ = op listtype  : ty -> ty
val _ = op pairtype  : ty * ty -> ty
val _ = op funtype   : ty list * ty -> ty
val _ = op asFuntype : ty -> (ty list * ty) option
(* definition of [[typeString]] for Hindley-Milner types S411d *)
fun typeString tau =
  case asFuntype tau
    of SOME (args, result) => 
         "(" ^ spaceSep (map typeString args) ^ " -> " ^ typeString result ^ ")"
     | NONE =>
         case tau
           of TYCON c => tyconString c
            | TYVAR a => a
            | CONAPP (tau, []) => "(" ^ typeString tau ^ ")"
            | CONAPP (tau, taus) =>
                "(" ^ typeString tau ^ " " ^ spaceSep (map typeString taus) ^
                                                                             ")"
(* shared utility functions on Hindley-Milner types 420 *)
type subst = ty env
fun varsubst theta = 
  (fn a => find (a, theta) handle NotFound _ => TYVAR a)
(* type declarations for consistency checking *)
type subst = subst
val _ = op varsubst : subst -> (name -> ty)
(* shared utility functions on Hindley-Milner types 421a *)
fun tysubst theta =
  let fun subst (TYVAR a) = varsubst theta a
        | subst (TYCON c) = TYCON c
        | subst (CONAPP (tau, taus)) = CONAPP (subst tau, map subst taus)
(* type declarations for consistency checking *)
val _ = op tysubst : subst -> (ty -> ty)
val _ = op subst   :           ty -> ty
  in  subst
  end
(* shared utility functions on Hindley-Milner types 421b *)
fun dom theta =
  map (fn (a, _) => a) theta
fun compose (theta2, theta1) =
  let val domain  = union (dom theta2, dom theta1)
      val replace = tysubst theta2 o varsubst theta1
  in  map (fn a => (a, replace a)) domain
  end
(* type declarations for consistency checking *)
val _ = op dom     : subst -> name set
val _ = op compose : subst * subst -> subst
(* shared utility functions on Hindley-Milner types ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
fun eqsubst (theta1, theta2) =
  let val domain  = union (dom theta2, dom theta1)
      fun eqOn a = varsubst theta1 a = varsubst theta2 a
  in  List.all eqOn domain
  end
(* shared utility functions on Hindley-Milner types 421c *)
fun instantiate (FORALL (formals, tau), actuals) =
  tysubst (bindList (formals, actuals, emptyEnv)) tau
  handle BindListLength =>
    raise BugInTypeInference "number of types in instantiation"
(* type declarations for consistency checking *)
val _ = op instantiate : type_scheme * ty list -> ty
(* shared utility functions on Hindley-Milner types 421d *)
infix 7 |-->
fun a |--> (TYVAR a') = if a = a' then emptyEnv
                        else bind (a, TYVAR a', emptyEnv)
  | a |--> tau        = bind (a, tau, emptyEnv)
(* type declarations for consistency checking *)
val _ = op |--> : name * ty -> subst
(* shared utility functions on Hindley-Milner types 422a *)
val idsubst = emptyEnv
(* shared utility functions on Hindley-Milner types 422b *)
fun eqType (TYVAR a, TYVAR a') = a = a'
  | eqType (TYCON c, TYCON c') = eqTycon (c, c')
  | eqType (CONAPP (tau, taus), CONAPP (tau', taus')) =
      eqType (tau, tau') andalso eqTypes (taus, taus')
  | eqType _ = false
and eqTypes (taus, taus') = ListPair.allEq eqType (taus, taus')
(* type declarations for consistency checking *)
val _ = op idsubst : subst
(* type declarations for consistency checking *)
val _ = op eqType : ty * ty -> bool
(* shared utility functions on Hindley-Milner types 444a *)
fun canonicalize (FORALL (bound, ty)) =
  let fun canonicalTyvarName n =
        if n < 26 then "'" ^ str (chr (ord #"a" + n))
        else "'v" ^ intString (n - 25)
      val free = diff (freetyvars ty, bound)
      fun unusedIndex n =
        if member (canonicalTyvarName n) free then unusedIndex (n+1) else n
      fun newBoundVars (index, [])                = []
        | newBoundVars (index, oldvar :: oldvars) =
            let val n = unusedIndex index
            in  canonicalTyvarName n :: newBoundVars (n+1, oldvars)
            end
      val newBound = newBoundVars (0, bound)
(* type declarations for consistency checking *)
val _ = op canonicalize : type_scheme -> type_scheme
val _ = op newBoundVars : int * name list -> name list
  in  FORALL (newBound,
              tysubst (bindList (bound, map TYVAR newBound, emptyEnv)) ty)
  end
(* shared utility functions on Hindley-Milner types 444b *)
local
  val n = ref 1
in
  fun freshtyvar _ = TYVAR ("'t" ^ intString (!n) before n := !n + 1)
(* type declarations for consistency checking *)
val _ = op freshtyvar : 'a -> ty
end
(* shared utility functions on Hindley-Milner types 445a *)
fun generalize (tau, tyvars) =
  canonicalize (FORALL (diff (freetyvars tau, tyvars), tau))
(* type declarations for consistency checking *)
val _ = op generalize : ty * name set -> type_scheme
(* shared utility functions on Hindley-Milner types 445b *)
fun freshInstance (FORALL (bound, tau)) =
  instantiate (FORALL (bound, tau), map freshtyvar bound)
(* type declarations for consistency checking *)
val _ = op freshInstance : type_scheme -> ty
(* shared utility functions on Hindley-Milner types S412b *)
fun typeSchemeString (FORALL ([], tau)) =
      typeString tau
  | typeSchemeString (FORALL (a's, tau)) =
      "(forall [" ^ spaceSep a's ^ "] " ^ typeString tau ^ ")"
(* type declarations for consistency checking *)
val _ = op typeString       : ty          -> string
val _ = op typeSchemeString : type_scheme -> string
(* shared utility functions on Hindley-Milner types ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
fun substString pairs =
      separate ("idsubst", " o ")
      (map (fn (a, t) => a ^ " |--> " ^ typeString t) pairs)
(* specialized environments for type schemes 446a *)
type type_env = type_scheme env * name set
(* specialized environments for type schemes 446b *)
val emptyTypeEnv = 
      (emptyEnv, emptyset)
fun findtyscheme (x, (Gamma, free)) = find (x, Gamma)
(* type declarations for consistency checking *)
val _ = op emptyTypeEnv : type_env
val _ = op findtyscheme : name * type_env -> type_scheme
(* specialized environments for type schemes 446c *)
fun bindtyscheme (x, sigma as FORALL (bound, tau), (Gamma, free)) = 
  (bind (x, sigma, Gamma), union (diff (freetyvars tau, bound), free))
(* specialized environments for type schemes 446d *)
fun freetyvarsGamma (_, free) = free



(*****************************************************************)
(*                                                               *)
(*   ABSTRACT SYNTAX AND VALUES FOR \NML                         *)
(*                                                               *)
(*****************************************************************)

(* abstract syntax and values for \nml S405c *)
(* definitions of [[exp]] and [[value]] for \nml 414 *)
datatype exp = LITERAL of value
             | VAR     of name
             | IFX     of exp * exp * exp
             | BEGIN   of exp list
             | APPLY   of exp * exp list
             | LETX    of let_kind * (name * exp) list * exp
             | LAMBDA  of name list * exp
and let_kind = LET | LETREC | LETSTAR
and (* definition of [[value]] for \nml 415b *)
    value = SYM       of name
          | NUM       of int
          | BOOLV     of bool
          | NIL
          | PAIR      of value * value
          | CLOSURE   of lambda * (unit -> value env)
          | PRIMITIVE of primop
    withtype primop = value list -> value (* raises RuntimeError *)
         and lambda = name list * exp
(* definition of [[def]] for \nml 415a *)
datatype def  = VAL    of name * exp
              | VALREC of name * exp
              | EXP    of exp
              | DEFINE of name * (name list * exp)
(* definition of [[unit_test]] for languages with Hindley-Milner types S405b *)
datatype unit_test = CHECK_EXPECT      of exp * exp
                   | CHECK_ASSERT      of exp
                   | CHECK_ERROR       of exp
                   | CHECK_TYPE        of exp * type_scheme
                   | CHECK_PTYPE       of exp * type_scheme
                   | CHECK_TYPE_ERROR  of def
(* definition of [[xdef]] (shared) S365b *)
datatype xdef = DEF    of def
              | USE    of name
              | TEST   of unit_test
              | DEFS   of def list  (*OMIT*)
(* definition of [[valueString]] for \uscheme, \tuscheme, and \nml 314 *)
fun valueString (SYM v)   = v
  | valueString (NUM n)   = intString n
  | valueString (BOOLV b) = if b then "#t" else "#f"
  | valueString (NIL)     = "()"
  | valueString (PAIR (car, cdr))  = 
      let fun tail (PAIR (car, cdr)) = " " ^ valueString car ^ tail cdr
            | tail NIL = ")"
            | tail v = " . " ^ valueString v ^ ")"
(* type declarations for consistency checking *)
val _ = op valueString : value -> string
      in  "(" ^ valueString car ^ tail cdr
      end
  | valueString (CLOSURE   _) = "<function>"
  | valueString (PRIMITIVE _) = "<function>"
(* definition of [[expString]] for \nml\ and \uml S417a *)
fun expString e =
  let fun bracket s = "(" ^ s ^ ")"
      fun sqbracket s = "[" ^ s ^ "]"
      val bracketSpace = bracket o spaceSep
      fun exps es = map expString es
      fun withBindings (keyword, bs, e) =
        bracket (spaceSep [keyword, bindings bs, expString e])
      and bindings bs = bracket (spaceSep (map binding bs))
      and binding (x, e) = sqbracket (x ^ " " ^ expString e)
      val letkind = fn LET => "let" | LETSTAR => "let*" | LETREC => "letrec"
  in  case e
        of LITERAL v => valueString v
         | VAR name => name
         | IFX (e1, e2, e3) => bracketSpace ("if" :: exps [e1, e2, e3])
         | BEGIN es => bracketSpace ("begin" :: exps es)
         | APPLY (e, es) => bracketSpace (exps (e::es))
         | LETX (lk, bs, e) => bracketSpace [letkind lk, bindings bs, expString
                                                                              e]
         | LAMBDA (xs, body) => bracketSpace ["lambda",
                                              bracketSpace xs, expString body]
         (* extra cases of [[expString]] for \uml S417b *)
         (* this space is filled in by the uML appendix *)
  end
(* definitions of [[defString]] and [[defName]] for \nml\ and \uml S417c *)
fun defString d =
  let fun bracket s = "(" ^ s ^ ")"
      val bracketSpace = bracket o spaceSep
      fun formal (x, t) = "[" ^ x ^ " : " ^ typeString t ^ "]"
  in  case d
        of EXP e         => expString e
         | VAL    (x, e) => bracketSpace ["val",     x, expString e]
         | VALREC (x, e) => bracketSpace ["val-rec", x, expString e]
         | DEFINE (f, (formals, body)) =>
             bracketSpace ["define", f, bracketSpace formals, expString body]

(* cases for [[defString]] for forms found only in \uml ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
         (*empty*)
  end
fun defName (VAL (x, _))       = x
  | defName (VALREC (x, _)) = x
  | defName (DEFINE (x, _)) = x
  | defName (EXP _) = raise InternalError "asked for name defined by expression"

(* clauses for [[defName]] for forms found only in \uml ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
  (*empty*)


(*****************************************************************)
(*                                                               *)
(*   UTILITY FUNCTIONS ON \USCHEME, \TUSCHEME, AND \NML\ VALUES  *)
(*                                                               *)
(*****************************************************************)

(* utility functions on \uscheme, \tuscheme, and \nml\ values 315a *)
fun embedInt n = NUM n
fun projectInt (NUM n) = n
  | projectInt v =
      raise RuntimeError ("value " ^ valueString v ^ " is not an integer")
(* type declarations for consistency checking *)
val _ = op embedInt   : int   -> value
val _ = op projectInt : value -> int
(* utility functions on \uscheme, \tuscheme, and \nml\ values 315b *)
fun embedBool b = BOOLV b
fun projectBool (BOOLV false) = false
  | projectBool _             = true
(* type declarations for consistency checking *)
val _ = op embedBool   : bool  -> value
val _ = op projectBool : value -> bool
(* utility functions on \uscheme, \tuscheme, and \nml\ values 315c *)
fun embedList []     = NIL
  | embedList (h::t) = PAIR (h, embedList t)
(* utility functions on \uscheme, \tuscheme, and \nml\ values S365d *)
fun equalatoms (NIL,      NIL    )  = true
  | equalatoms (NUM  n1,  NUM  n2)  = n1 = n2
  | equalatoms (SYM  v1,  SYM  v2)  = v1 = v2
  | equalatoms (BOOLV b1, BOOLV b2) = b1 = b2
  | equalatoms  _                   = false
(* type declarations for consistency checking *)
val _ = op equalatoms : value * value -> bool
(* utility functions on \uscheme, \tuscheme, and \nml\ values S366a *)
fun equalpairs (PAIR (car1, cdr1), PAIR (car2, cdr2)) =
      equalpairs (car1, car2) andalso equalpairs (cdr1, cdr2)
  | equalpairs (v1, v2) = equalatoms (v1, v2)
(* type declarations for consistency checking *)
val _ = op equalpairs : value * value -> bool
(* utility functions on \uscheme, \tuscheme, and \nml\ values S366b *)
val testEqual = equalpairs
(* type declarations for consistency checking *)
val _ = op testEqual : value * value -> bool
(* utility functions on \uscheme, \tuscheme, and \nml\ values S379 *)
fun cycleThrough xs =
  let val remaining = ref xs
      fun next () = case !remaining
                      of [] => (remaining := xs; next ())
                       | x :: xs => (remaining := xs; x)
  in  if null xs then
        raise InternalError "empty list given to cycleThrough"
      else
        next
  end
val unspecified =
  cycleThrough [BOOLV true, NUM 39, SYM "this value is unspecified", NIL,
                PRIMITIVE (fn _ => let exception Unspecified in raise
                                                               Unspecified end)]
(* type declarations for consistency checking *)
val _ = op cycleThrough : 'a list -> (unit -> 'a)
val _ = op unspecified  : unit -> value
(* utility functions on \uscheme, \tuscheme, and \nml\ values S409c *)
fun primitiveEquality (v, v') =
 let fun noFun () = raise RuntimeError "compared functions for equality"
 in  case (v, v')
       of (NIL,      NIL    )  => true
        | (NUM  n1,  NUM  n2)  => n1 = n2
        | (SYM  v1,  SYM  v2)  => v1 = v2
        | (BOOLV b1, BOOLV b2) => b1 = b2
        | (PAIR (v, vs), PAIR (v', vs')) =>
            primitiveEquality (v, v') andalso primitiveEquality (vs, vs')
        | (PAIR _,   NIL)    => false
        | (NIL,      PAIR _) => false
        | (CLOSURE   _, _) => noFun ()
        | (PRIMITIVE _, _) => noFun ()
        | (_, CLOSURE   _) => noFun ()
        | (_, PRIMITIVE _) => noFun ()
        | _ => raise BugInTypeInference
                       ("compared incompatible values " ^ valueString v ^
                                                                       " and " ^
                        valueString v' ^ " for equality")
 end
(* utility functions on \uscheme, \tuscheme, and \nml\ values 390b *)
val unitVal = NIL
(* type declarations for consistency checking *)
val _ = op unitVal : value



(*****************************************************************)
(*                                                               *)
(*   TYPE INFERENCE FOR \NML\ AND \UML                           *)
(*                                                               *)
(*****************************************************************)

(* type inference for \nml\ and \uml S405e *)
(* representation of type constraints 446e *)
datatype con = ~  of ty  * ty
             | /\ of con * con
             | TRIVIAL
infix 4 ~
infix 3 /\
(* utility functions on type constraints 447a *)
fun freetyvarsConstraint (t ~  t') = union (freetyvars t, freetyvars t')
  | freetyvarsConstraint (c /\ c') = union (freetyvarsConstraint c,
                                             freetyvarsConstraint c')
  | freetyvarsConstraint TRIVIAL    = emptyset
(* utility functions on type constraints 447b *)
fun consubst theta =
  let fun subst (tau1 ~ tau2) = tysubst theta tau1 ~ tysubst theta tau2
        | subst (c1 /\ c2)    = subst c1 /\ subst c2
        | subst TRIVIAL       = TRIVIAL
  in  subst
  end
(* type declarations for consistency checking *)
val _ = op bindtyscheme : name * type_scheme * type_env -> type_env
(* type declarations for consistency checking *)
val _ = op freetyvarsGamma : type_env -> name set
(* type declarations for consistency checking *)
val _ = op consubst : subst -> con -> con
(* utility functions on type constraints 447c *)
fun conjoinConstraints []      = TRIVIAL
  | conjoinConstraints [c]     = c
  | conjoinConstraints (c::cs) = c /\ conjoinConstraints cs
(* type declarations for consistency checking *)
val _ = op conjoinConstraints : con list -> con
(* utility functions on type constraints 448b *)
fun isSolved TRIVIAL = true
  | isSolved (tau ~ tau') = eqType (tau,tau')
  | isSolved (c /\ c') = isSolved c andalso isSolved c'
fun solves (theta, c) = isSolved (consubst theta c)
(* type declarations for consistency checking *)
val _ = op isSolved : con -> bool
val _ = op solves : subst * con -> bool
(* utility functions on type constraints S405d *)
(* definitions of [[constraintString]] and [[untriviate]] S412a *)
fun constraintString (c /\ c') = constraintString c ^ " /\\ " ^ constraintString
                                                                              c'
  | constraintString (t ~  t') = typeString t ^ " ~ " ^ typeString t'
  | constraintString TRIVIAL = "TRIVIAL"

fun untriviate (c /\ c') = (case (untriviate c, untriviate c')
                              of (TRIVIAL, c) => c
                               | (c, TRIVIAL) => c
                               | (c, c') => c /\ c')
  | untriviate atomic = atomic
(* type declarations for consistency checking *)
val _ = op constraintString : con -> string
val _ = op untriviate       : con -> con
(* constraint solving 447d *)
fun unsatisfiableEquality (t1, t2) =
  let val t1_arrow_t2 = funtype ([t1], t2)
      val FORALL (_, canonical) =
            canonicalize (FORALL (freetyvars t1_arrow_t2, t1_arrow_t2))
  in  case asFuntype canonical
        of SOME ([t1'], t2') =>
             raise TypeError ("cannot make " ^ typeString t1' ^
                              " equal to " ^ typeString t2')
         | _ => let exception Can'tHappen in raise Can'tHappen end
  end
(* constraint solving ((prototype)) 448a *)
fun solve (TYVAR t1 ~ TYVAR t2)                 = t1 |--> TYVAR t2
  | solve (TYVAR t1 ~ TYCON t2)                 = t1 |--> TYCON  t2
  | solve (TYVAR t1 ~ CONAPP t2)                = 
     let val conappFrees  = freetyvars (CONAPP t2)
     in if not (member t1 conappFrees) then
          t1 |--> CONAPP t2
        else
          unsatisfiableEquality (TYVAR t1, CONAPP t2)
     end
  | solve (TYCON t1 ~ TYVAR t2)                 = solve (TYVAR t2 ~ TYCON t1)
  | solve (TYCON t1 ~ TYCON t2)                 = 
     if t1 = t2 then
       idsubst
     else
       unsatisfiableEquality (TYCON t1, TYCON t2)
  | solve (TYCON t1 ~ CONAPP t2)                = 
     unsatisfiableEquality (TYCON t1, CONAPP t2)
  | solve (CONAPP t1 ~ TYVAR t2)                = solve (TYVAR t2 ~ CONAPP t1)
  | solve (CONAPP t1 ~ TYCON t2)                = 
     unsatisfiableEquality (CONAPP t1, TYCON t2)
  | solve (CONAPP (t1, t1s) ~ CONAPP (t2, t2s)) = 
     let val c1 = ListPair.foldrEq (fn (x, y, z) => 
       (x ~ y) /\ z) (t1 ~ t2) (t1s,t2s) handle UnequalLengths => 
         unsatisfiableEquality (CONAPP (t1, t1s), CONAPP (t2, t2s))
     in solve c1
     end
  | solve (c1/\c2)                              = 
    let val theta1            = solve c1
        val c2prime           = consubst theta1 c2
        val theta2            = solve c2prime
        val thetasComp        = compose (theta2, theta1)
    in thetasComp
    end
  | solve (TRIVIAL) = idsubst


(*Spec unit tests*)
fun isIdempotent pairs =
    let fun distinct a' (a, tau) = 
      a <> a' andalso not (member a' (freetyvars tau))
        fun good (prev', (a, tau)::next) =
              List.all (distinct a) prev' andalso List.all (distinct a) next
              andalso good ((a, tau)::prev', next)
          | good (_, []) = true
    in  good ([], pairs)
    end


val solve =
    fn c => let val theta = solve c
            in  if isIdempotent theta then theta
                else raise BugInTypeInference "non-idempotent substitution"
            end

(* type declarations for consistency checking *)
val _ = op solve : con -> subst
(* constraint solving ((elided)) (THIS CAN'T HAPPEN -- claimed code was not used) *)
fun hasNoSolution c = (solve c; false) handle TypeError _ => true
fun hasGoodSolution c = solves (solve c, c) handle TypeError _ => false
val hasSolution = not o hasNoSolution : con -> bool
fun solutionEquivalentTo (c, theta) = eqsubst (solve c, theta)

(*unit tests*)
val () =
  let val c1 = TYVAR "a" ~ TYVAR "b"
      val c2 = TYVAR "a" ~ TYVAR "c"
      val testname = constraintString (c1 /\ c2)
  in Unit.checkAssert testname
    (fn () => hasGoodSolution (c1 /\ c2) )
  end

val () =
  let val c1 = TYVAR "a" ~ inttype
      val c2 = TYVAR "a" ~ booltype
      val testname = constraintString (c1 /\ c2)
  in Unit.checkAssert testname
    (fn () => hasNoSolution (c1 /\ c2) )
  end

val () =
  let val c1 = inttype ~  TYVAR "a"
      val testname = "has good solution tycon tyvar"
  in Unit.checkAssert testname
    (fn () => hasGoodSolution c1)
  end

val () =
  let val c1 = booltype ~ inttype
      val testname = "has no solution tycon tycon"
  in Unit.checkAssert testname
    (fn () => hasNoSolution c1)
  end

val () =
  let val c1 = booltype ~ booltype
      val testname = "has good solution tycon tycon"
  in Unit.checkAssert testname
    (fn () => hasGoodSolution c1)
  end

val () =
  let val c1 = CONAPP (TYVAR "list", [inttype])  ~ booltype
      val testname = "has no solution conapp tyvar"
  in Unit.checkAssert testname
    (fn () => hasNoSolution c1)
  end

val () =
  let val c1 =  booltype ~ CONAPP (TYVAR "list", [inttype])
      val testname = "has no solution tyvar conapp"
  in Unit.checkAssert testname
    (fn () => hasNoSolution c1)
  end

val () =
  let val c1 =  TYVAR "a" ~ CONAPP (TYVAR "list", [TYVAR "a"])
      val testname = "has no solution tyvar conapp"
  in Unit.checkAssert testname
    (fn () => hasNoSolution c1)
  end

val () =
  let val c1 =  TYVAR "b" ~ CONAPP (TYVAR "list", [TYVAR "a"])
      val testname = "has no solution tyvar conapp"
  in Unit.checkAssert testname
    (fn () => hasGoodSolution c1)
  end

val () =
  let val c1 = 
    CONAPP (TYVAR "list", [inttype]) ~ CONAPP (TYVAR "list", [TYVAR "a"])
      val testname = "has no solution tyvar conapp"
  in Unit.checkAssert testname
    (fn () => hasGoodSolution c1)
  end

val () =
  let val c1 =  
    CONAPP (TYVAR "list", [inttype]) ~ CONAPP (TYVAR "list", [booltype])
      val testname = "has no solution tyvar conapp"
  in Unit.checkAssert testname
    (fn () => hasNoSolution c1)
  end




(* exhaustiveness analysis for {\uml} S419f *)
(* filled in when implementing uML *)
(* definitions of [[typeof]] and [[typdef]] for \nml\ and \uml 448c *)
fun typeof (e, Gamma) =
  let
(* shared definition of [[typesof]], to infer the types of a list of expressions 448d *)
      fun typesof ([],    Gamma) = ([], TRIVIAL)
        | typesof (e::es, Gamma) =
            let val (tau,  c)  = typeof  (e,  Gamma)
                val (taus, c') = typesof (es, Gamma)
            in  (tau :: taus, c /\ c')
            end

(* function [[literal]], to infer the type of a literal constant ((prototype)) 448e *)
      fun literal (l) = 
          (case l
            of (BOOLV x)       => (booltype, TRIVIAL)
            | (NUM x)          => (inttype, TRIVIAL)
            | (SYM x)          => (symtype, TRIVIAL)
            | (NIL)            => 
               let val a = freshtyvar "a"
               in (listtype a, TRIVIAL)
               end
            | (PAIR (v1, v2))  => 
               let val (tauv1, conv1) = literal v1
                   val (tauv2, conv2) = literal v2
                   val conv3          = tauv2 ~ listtype tauv1
               in (tauv2,conv3 /\ conv1 /\ conv2)
               end
            | (CLOSURE (x, y)) => raise BugInTypeInference "closure"
            | (PRIMITIVE x)    => raise BugInTypeInference "closure")

(* function [[ty]], to infer the type of a \nml\ expression, given [[Gamma]] 449a *)
      fun ty (LITERAL n) = literal n
        (* more alternatives for [[ty]] 449b *)
        | ty (VAR x) = (freshInstance (findtyscheme (x, Gamma)), TRIVIAL)
        (* more alternatives for [[ty]] 449c *)
        | ty (APPLY (f, actuals)) = 
             (case typesof (f :: actuals, Gamma)
                of ([], _) => let exception Can'tHappen in raise Can'tHappen end
                 | (funty :: actualtypes, c) => 
                     let val rettype = freshtyvar ()
                     in (rettype, c /\ (funty ~ 
                                        funtype (actualtypes, rettype)))
                     end)
        (* more alternatives for [[ty]] 449d *)
        | ty (LETX (LETSTAR, [], body)) = ty body
        | ty (LETX (LETSTAR, (b :: bs), body)) = 
            ty (LETX (LET, [b], LETX (LETSTAR, bs, body)))
        (* more alternatives for [[ty]] ((prototype)) 449e *)
        | ty (IFX (e1, e2, e3))        = 
           let val typesofIFX = typesof ([e1, e2, e3], Gamma)
               val tysIFX     = fst typesofIFX
               val c1IFX      = snd typesofIFX
               val t1         = List.nth (tysIFX, 0)
               val t2         = List.nth (tysIFX, 1)
               val t3         = List.nth (tysIFX, 2)
               val c2IFX      = c1IFX /\ (t1 ~ booltype) /\ (t2 ~ t3)
           in (t2, c2IFX)
           end
        | ty (BEGIN es)                = 
          (case es
            of ([]) => (unittype, TRIVIAL)
            | _     => 
              let val typesofBEGIN = typesof (es, Gamma)
                  val tysBEGIN     = fst typesofBEGIN
                  val c1BEGIN      = snd typesofBEGIN
                  val tn           = List.last tysBEGIN 
              in  (tn ,c1BEGIN)
              end)
        | ty (LAMBDA (formals, body))  = 
           let val alphas            = map freshtyvar formals
               val newGamma          = 
                 ListPair.foldr (fn (x, y, z) => 
                   bindtyscheme(x, FORALL ([], y), z)) Gamma (formals, alphas)
               val (funReturnTy, c1) = typeof (body, newGamma)
               val finalty           = funtype(alphas, funReturnTy) 
           in (finalty, c1)
           end
        | ty (LETX (LET, bs, body))    = 
           let val names             = map fst bs
               val exps              = map snd bs     
               val (tysLet, c1LET )  = typesof (exps, Gamma)
               val thetac1           = solve c1LET
               val alphas            = dom thetac1
               val ftv               = freetyvarsGamma Gamma
               val goodalphas        = inter (alphas, ftv)
               val goodalphasTyForm  = List.map (fn (x) => TYVAR x) goodalphas 
               val sub               = tysubst thetac1
               val goodalphasChecked = 
                 List.map (fn (x) => sub x) goodalphasTyForm
               val alphasConList     = 
                 ListPair.map (fn (x, y) => 
                   x ~ y) (goodalphasTyForm, goodalphasChecked)
               val cprime            = conjoinConstraints alphasConList
               val cprimefrees       = freetyvarsConstraint cprime
               val cprimeGammafrees  = union (ftv, cprimefrees)
               val typeschemes       = 
                 List.map (fn (x) => 
                   generalize (sub x, cprimeGammafrees)) tysLet
               val newGamma          = 
                 ListPair.foldr (fn (x, y, z) => 
                   bindtyscheme(x, y, z)) Gamma (names, typeschemes)
               val (returnTy, cb)    = typeof (body, newGamma)
           in  (returnTy, cprime /\ cb) 
           end
        | ty (LETX (LETREC, bs, body)) =
           let val names                  = map fst bs
               val exps                   = map snd bs 
               val alphas                 = map freshtyvar names 
               val newGamma               = 
                 ListPair.foldr (fn (x, y, z) => 
                   bindtyscheme(x, FORALL ([], y), z)) Gamma (names, alphas)
               val (tysLetrec, c1LETREC ) = typesof (exps, newGamma) 
               val alphasConList          = 
                 ListPair.map (fn (x, y) => x ~ y) (tysLetrec, alphas) 
               val c2                     = conjoinConstraints alphasConList
               val c3                     = c2 /\ c1LETREC
               val thetac1                = solve c3
               val alphas                 = dom thetac1
               val ftv                    = freetyvarsGamma Gamma
               val goodalphas             = inter (alphas, ftv)
               val goodalphasTyForm       = 
                 List.map (fn (x) => TYVAR x) goodalphas 
               val sub                    = tysubst thetac1
               val goodalphasChecked      = 
                 List.map (fn (x) => sub x) goodalphasTyForm
               val alphasConList          = 
                 ListPair.map (fn (x, y) => 
                   x ~ y) (goodalphasTyForm, goodalphasChecked)
               val cprime                 = conjoinConstraints alphasConList
               val cprimefrees            = freetyvarsConstraint cprime
               val cprimeGammafrees       = union (ftv, cprimefrees)
               val typeschemes            = 
                 List.map (fn (x) => 
                   generalize (sub x, cprimeGammafrees)) tysLetrec
               val newGamma               = 
                 ListPair.foldr (fn (x, y, z) => 
                   bindtyscheme(x, y, z)) Gamma (names, typeschemes)
               val (returnTy, cb)         = typeof (body, newGamma)
           in  (returnTy, cprime /\ cb) 
           end
(* type declarations for consistency checking *)
val _ = op typeof  : exp      * type_env -> ty      * con
val _ = op typesof : exp list * type_env -> ty list * con
val _ = op literal : value -> ty * con
val _ = op ty      : exp   -> ty * con
  in  ty e
  end
(* definitions of [[typeof]] and [[typdef]] for \nml\ and \uml 449f *)
fun typdef (d, Gamma) =
  case d
    of VAL    (x, e)      =>
                   (* infer and bind type for [[VAL    (x, e)]] for \nml 450a *)
                             let val (tau, c) = typeof (e, Gamma)
                                 val theta    = solve c
                                 val sigma    = generalize (tysubst theta tau,
                                                          freetyvarsGamma Gamma)
                             in  (bindtyscheme (x, sigma, Gamma),
                                                         typeSchemeString sigma)
                             end
     | VALREC (x, e)      =>
                   (* infer and bind type for [[VALREC (x, e)]] for \nml 450b *)
                             let val alpha    = freshtyvar ()
                                 val Gamma'   = bindtyscheme (x, FORALL ([],
                                                                  alpha), Gamma)
                                 val (tau, c) = typeof (e, Gamma')
                                 val theta    = solve (c /\ alpha ~ tau)
                                 val sigma    = generalize (tysubst theta alpha,
                                                          freetyvarsGamma Gamma)
                                 val ()       =
                   (* if [[e]] is not a [[LAMBDA]], raise [[TypeError]] S405a *)
                                                case e of LAMBDA _ => ()
                                                        | _ => raise TypeError (
                                  "in val-rec, right-hand side " ^ expString e ^

                                                             " is not a lambda")
                             in  (bindtyscheme (x, sigma, Gamma),
                                                         typeSchemeString sigma)
                             end
     | EXP e              => typdef (VAL ("it", e), Gamma)
     | DEFINE (x, lambda) => typdef (VALREC (x, LAMBDA lambda), Gamma)
     (* extra case for [[typdef]] used only in \uml S419e *)
     (* filled in when implementing uML *)
(* type declarations for consistency checking *)
val _ = op typdef : def * type_env -> type_env * string
(* type declarations for consistency checking *)
val _ = op typeof  : exp * type_env -> ty * con
val _ = op typdef : def * type_env -> type_env * string



(*****************************************************************)
(*                                                               *)
(*   LEXICAL ANALYSIS AND PARSING FOR \NML, PROVIDING [[FILEXDEFS]] AND [[STRINGSXDEFS]] *)
(*                                                               *)
(*****************************************************************)

(* lexical analysis and parsing for \nml, providing [[filexdefs]] and [[stringsxdefs]] S412c *)
(* lexical analysis for \uscheme\ and related languages S373c *)
datatype pretoken = QUOTE
                  | INT     of int
                  | SHARP   of bool
                  | NAME    of string
type token = pretoken plus_brackets
(* lexical analysis for \uscheme\ and related languages S373d *)
fun pretokenString (QUOTE)     = "'"
  | pretokenString (INT  n)    = intString n
  | pretokenString (SHARP b)   = if b then "#t" else "#f"
  | pretokenString (NAME x)    = x
val tokenString = plusBracketsString pretokenString
(* lexical analysis for \uscheme\ and related languages S374a *)
local
  (* functions used in all lexers S374c *)
  fun noneIfLineEnds chars =
    case streamGet chars
      of NONE => NONE (* end of line *)
       | SOME (#";", cs) => NONE (* comment *)
       | SOME (c, cs) => 
           let val msg = "invalid initial character in `" ^
                         implode (c::listOfStream cs) ^ "'"
           in  SOME (ERROR msg, EOS)
           end
  (* type declarations for consistency checking *)
  val _ = op noneIfLineEnds : 'a lexer
  (* functions used in the lexer for \uscheme S374b *)
  fun atom "#t" = SHARP true
    | atom "#f" = SHARP false
    | atom x    = NAME x
in
  val schemeToken =
    whitespace *>
    bracketLexer   (  QUOTE   <$  eqx #"'" one
                  <|> INT     <$> intToken isDelim
                  <|> (atom o implode) <$> many1 (sat (not o isDelim) one)
                  <|> noneIfLineEnds
                   )
(* type declarations for consistency checking *)
val _ = op schemeToken : token lexer
val _ = op atom : string -> pretoken
end
(* parsers for single \uscheme\ tokens S374d *)
type 'a parser = (token, 'a) polyparser
val pretoken  = (fn (PRETOKEN t)=> SOME t  | _ => NONE) <$>? token : pretoken
                                                                          parser
val quote     = (fn (QUOTE)     => SOME () | _ => NONE) <$>? pretoken
val int       = (fn (INT   n)   => SOME n  | _ => NONE) <$>? pretoken
val booltok   = (fn (SHARP b)   => SOME b  | _ => NONE) <$>? pretoken
val name      = (fn (NAME  n)   => SOME n  | _ => NONE) <$>? pretoken
val any_name  = name
(* parsers for \nml\ tokens S413b *)
val arrow = eqx "->" name
val name  = sat (fn n => n <> "->") name  (* an arrow is not a name *)
val tyvar = quote *> (curry op ^ "'" <$> name <?>
                                               "type variable (got quote mark)")
(* parsers and parser builders for formal parameters and bindings S375a *)
fun formalsOf what name context = 
  nodups ("formal parameter", context) <$>! @@ (bracket (what, many name))

fun bindingsOf what name exp =
  let val binding = bracket (what, pair <$> name <*> exp)
  in  bracket ("(... " ^ what ^ " ...) in bindings", many binding)
  end

fun distinctBsIn bindings context =
  let fun check (loc, bs) =
        nodups ("bound name", context) (loc, map fst bs) >>=+ (fn _ => bs)
  in  check <$>! @@ bindings
  end
(* type declarations for consistency checking *)
val _ = op formalsOf  : string -> name parser -> string -> name list parser
val _ = op bindingsOf : string -> 'x parser -> 'e parser -> ('x * 'e) list
                                                                          parser
val _ = op distinctBsIn : (name * 'e) list parser -> string -> (name * 'e) list
                                                                          parser
(* parsers and parser builders for formal parameters and bindings S375b *)
fun recordFieldsOf name =
  nodups ("record fields", "record definition") <$>!
                                    @@ (bracket ("(field ...)", many name))
(* type declarations for consistency checking *)
val _ = op recordFieldsOf : name parser -> name list parser
(* parsers and parser builders for formal parameters and bindings S375c *)
fun kw keyword = 
  eqx keyword any_name
fun usageParsers ps = anyParser (map (usageParser kw) ps)
(* type declarations for consistency checking *)
val _ = op kw : string -> string parser
val _ = op usageParsers : (string * 'a parser) list -> 'a parser
(* parsers and parser builders for \scheme-like syntax S375d *)
fun sexp tokens = (
     SYM       <$> (notDot <$>! @@ any_name)
 <|> NUM       <$> int
 <|> embedBool <$> booltok
 <|> leftCurly <!> "curly brackets may not be used in S-expressions"
 <|> embedList <$> bracket ("list of S-expressions", many sexp)
 <|> (fn v => embedList [SYM "quote", v]) 
               <$> (quote *> sexp)
) tokens
and notDot (loc, ".") =
      errorAt "this interpreter cannot handle . in quoted S-expressions" loc
  | notDot (_,   s)   = OK s
(* type declarations for consistency checking *)
val _ = op sexp : value parser
(* parsers and parser builders for \scheme-like syntax S376a *)
fun atomicSchemeExpOf name =  VAR                   <$> name
                          <|> LITERAL <$> NUM       <$> int
                          <|> LITERAL <$> embedBool <$> booltok
(* parsers and parser builders for \scheme-like syntax S376d *)
fun fullSchemeExpOf atomic keywordsOf =
  let val exp = fn tokens => fullSchemeExpOf atomic keywordsOf tokens
  in      atomic
      <|> keywordsOf exp
      <|> quote *> (LITERAL <$> sexp)
      <|> quote *> badRight "quote ' followed by right bracket"
      <|> leftCurly <!> "curly brackets are not supported"
      <|> left *> right <!> "(): unquoted empty parentheses"
      <|> bracket("function application", curry APPLY <$> exp <*> many exp)
  end
(* parser builders for typed languages S395e *)
val distinctTyvars = 
  nodups ("quantified type variable", "forall") <$>! @@ (many tyvar)

fun arrowsOf conapp funty =
  let fun arrows []              [] = ERROR "empty type ()"
        | arrows (tycon::tyargs) [] = OK (conapp (tycon, tyargs))
        | arrows args            [rhs] =
            (case rhs of [result] => OK (funty (args, result))
                       | []       => ERROR "no result type after function arrow"
                       | _        => ERROR
                                   "multiple result types after function arrow")
        | arrows args (_::_::_) = ERROR "multiple arrows in function type"
  in  fn xs => errorLabel "syntax error: " o arrows xs
  end
(* parsers for Hindley-Milner types with named type constructors S413c *)
val arrows = arrowsOf CONAPP funtype

fun ty tokens = (
     TYCON <$> sat (curry op <> "->") any_name
 <|> TYVAR <$> tyvar
 <|> usageParsers [("(forall (tyvars) type)", bracket ("('a ...)", many tyvar)
                                                                         *> ty)]
     <!> "nested 'forall' type is not a Hindley-Milner type"
 <|> bracket ("constructor application",
              arrows <$> many ty <*>! many (arrow *> many ty))
) tokens

val tyscheme =
      usageParsers [("(forall (tyvars) type)",
                     curry FORALL <$> bracket ("['a ...]", distinctTyvars) <*>
                                                                            ty)]
  <|> curry FORALL [] <$> ty
  <?> "type"
(* type declarations for consistency checking *)
val _ = op tyvar : string parser
val _ = op ty    : ty     parser
(* parsers and [[xdef]] streams for \nml S412d *)
fun exptable exp =
  let val bindings = bindingsOf "(x e)" name exp
      val dbs      = distinctBsIn bindings
      fun letx kind bs exp = LETX (kind, bs, exp)
      val formals = formalsOf "(x1 x2 ...)" name "lambda"
(* type declarations for consistency checking *)
val _ = op exp      : exp parser
val _ = op exptable : exp parser -> exp parser
  in usageParsers
     [ ("(if e1 e2 e3)",            curry3 IFX          <$> exp <*> exp <*> exp)
     , ("(begin e1 ...)",                  BEGIN        <$> many exp)
     , ("(lambda (names) body)",    curry  LAMBDA       <$> formals      <*> exp
                                                                               )
     , ("(let (bindings) body)",    curry3 LETX LET     <$> dbs "let"    <*> exp
                                                                               )
     , ("(letrec (bindings) body)", curry3 LETX LETREC  <$> dbs "letrec" <*> exp
                                                                               )
     , ("(let* (bindings) body)",   curry3 LETX LETSTAR <$> bindings     <*> exp
                                                                               )
     (* rows added to \nml's [[exptable]] in exercises S413a *)
     (* add syntactic extensions here, each preceded by a comma *)
     ]
  end

val exp = fullSchemeExpOf (atomicSchemeExpOf name) exptable
(* parsers and [[xdef]] streams for \nml S413d *)
val deftable = usageParsers
  [ ("(define f (args) body)",
                  let val formals = formalsOf "(x1 x2 ...)" name "define"
                  in  curry DEFINE <$> name <*> (pair <$> formals <*> exp)
                  end)
  , ("(val x e)",     curry VAL    <$> name <*> exp)
  , ("(val-rec x e)", curry VALREC <$> name <*> exp)
  ]

val testtable = usageParsers
  [ ("(check-expect e1 e2)",         curry CHECK_EXPECT <$> exp <*> exp)
  , ("(check-assert e)",                   CHECK_ASSERT <$> exp)
  , ("(check-error e)",                    CHECK_ERROR  <$> exp)
  , ("(check-type e tau)",           curry CHECK_TYPE   <$> exp <*> tyscheme)
  , ("(check-principal-type e tau)", curry CHECK_PTYPE  <$> exp <*> tyscheme)
  , ("(check-type-error e)",               CHECK_TYPE_ERROR <$> (deftable <|>
                                                                 EXP <$> exp))
  ]

val xdeftable = usageParsers
  [ ("(use filename)", USE <$> name)
  (* rows added to \nml's [[xdeftable]] in exercises S414a *)
  (* add syntactic extensions here, each preceded by a comma *)
  ]

val xdef  =  TEST <$> testtable
         <|> DEF  <$> deftable
         <|>          xdeftable
         <|> badRight "unexpected right bracket"
         <|> DEF <$> EXP <$> exp
         <?> "definition"
(* parsers and [[xdef]] streams for \nml S414b *)
val xdefstream = interactiveParsedStream (schemeToken, xdef)
(* shared definitions of [[filexdefs]] and [[stringsxdefs]] S254c *)
fun filexdefs (filename, fd, prompts) = xdefstream (filename, filelines fd,
                                                                        prompts)
fun stringsxdefs (name, strings) = xdefstream (name, streamOfList strings,
                                                                      noPrompts)
(* type declarations for consistency checking *)
val _ = op xdefstream   : string * line stream     * prompts -> xdef stream
val _ = op filexdefs    : string * TextIO.instream * prompts -> xdef stream
val _ = op stringsxdefs : string * string list               -> xdef stream



(*****************************************************************)
(*                                                               *)
(*   EVALUATION, TESTING, AND THE READ-EVAL-PRINT LOOP FOR \NML  *)
(*                                                               *)
(*****************************************************************)

(* evaluation, testing, and the read-eval-print loop for \nml S406a *)
(* definition of [[namedValueString]] for functional bridge languages S399c *)
fun namedValueString x v =
  case v of CLOSURE _ => x
          | PRIMITIVE _ => x
          | _ => valueString v
(* type declarations for consistency checking *)
val _ = op namedValueString : name -> value -> string
(* definitions of [[eval]] and [[evaldef]] for \nml\ and \uml S406b *)
fun eval (e, rho) =
  let val go = applyCheckingOverflow id in go end (* OMIT *)
  let fun ev (LITERAL v)        = v
        | ev (VAR x)            = find (x, rho)
        | ev (IFX (e1, e2, e3)) = ev (if projectBool (ev e1) then e2 else e3)
        | ev (LAMBDA l)         = CLOSURE (l, fn _ => rho)
        | ev (BEGIN es) =
            let fun b (e::es, lastval) = b (es, ev e)
                  | b (   [], lastval) = lastval
            in  b (es, embedBool false)
            end
        | ev (APPLY (f, args)) = 
           (case ev f
              of PRIMITIVE prim => prim (map ev args)
               | CLOSURE clo =>
                            (* apply closure [[clo]] to [[args]] ((ml)) S406c *)
                                let val ((formals, body), mkRho) = clo
                                    val actuals = map ev args
                                in  eval (body, bindList (formals, actuals,
                                                                      mkRho ()))
                                    handle BindListLength => 
                                        raise BugInTypeInference
                                          "Wrong number of arguments to closure"
                                end
               | _ => raise BugInTypeInference "Applied non-function"
               )
        (* more alternatives for [[ev]] for \nml\ and \uml S407a *)
        | ev (LETX (LET, bs, body)) =
            let val (names, values) = ListPair.unzip bs
            in  eval (body, bindList (names, map ev values, rho))
            end
        (* more alternatives for [[ev]] for \nml\ and \uml S407b *)
        | ev (LETX (LETSTAR, bs, body)) =
            let fun step ((x, e), rho) = bind (x, eval (e, rho), rho)
            in  eval (body, foldl step rho bs)
            end
        (* more alternatives for [[ev]] for \nml\ and \uml S407c *)
        | ev (LETX (LETREC, bs, body)) =
            let fun makeRho' () =
                  let fun step ((x, e), rho) =
                            (case e
                               of LAMBDA l => bind (x, CLOSURE (l, makeRho'),
                                                                            rho)
                                | _ => raise BugInTypeInference
                                                         "non-lambda in letrec")
                  in  foldl step rho bs
                  end
            in  eval (body, makeRho'())
            end
  in  ev e
  end
(* type declarations for consistency checking *)
val _ = op eval : exp * value env -> value
(* definitions of [[eval]] and [[evaldef]] for \nml\ and \uml S407d *)
fun evaldef (VAL (x, e), rho) =
      let val v   = eval (e, rho)
          val rho = bind (x, v, rho)
      in  (rho, namedValueString x v)
      end
  | evaldef (VALREC (f, LAMBDA lambda), rho) =
      let fun makeRho' () = bind (f, CLOSURE (lambda, makeRho'), rho)
          val v           = CLOSURE (lambda, makeRho')
      in  (makeRho'(), f)
      end
  | evaldef (VALREC _, rho) =
      raise BugInTypeInference "expression in val-rec is not lambda"
  | evaldef (EXP e, rho) = 
      let val v   = eval (e, rho)
          val rho = bind ("it", v, rho)
      in  (rho, valueString v)
      end
(* definitions of [[eval]] and [[evaldef]] for \nml\ and \uml S408a *)
  | evaldef (DEFINE (f, lambda), rho) =
      evaldef (VALREC (f, LAMBDA lambda), rho)
  (* clause for [[evaldef]] for datatype definition (\uml\ only) S408b *)
  (* code goes here in Chapter 11 *)
(* type declarations for consistency checking *)
val _ = op evaldef : def * value env -> value env * string
(* definitions of [[basis]] and [[processDef]] for \nml S410b *)
type basis = type_env * value env
fun processDef (d, (Gamma, rho), interactivity) =
  let val (Gamma, tystring)  = typdef  (d, Gamma)
      val (rho,   valstring) = evaldef (d, rho)
      val _ = if prints interactivity then
                println (valstring ^ " : " ^ tystring)
              else
                ()
(* type declarations for consistency checking *)
val _ = op processDef : def * basis * interactivity -> basis
  in  (Gamma, rho)
  end
fun dump_names (types, values) = app (println o fst) values  (*OMIT*)
(* shared definition of [[withHandlers]] S371a *)
fun withHandlers f a caught =
  f a
  handle RuntimeError msg   => caught ("Run-time error <at loc>: " ^ msg)
       | NotFound x         => caught ("Name " ^ x ^ " not found <at loc>")
       | Located (loc, exn) =>
           withHandlers (fn _ => raise exn) a (fn s => caught (fillAtLoc (s, loc
                                                                             )))

(* other handlers that catch non-fatal exceptions and pass messages to [[caught]] S371b *)
       | Div                => caught "Division by zero <at loc>"
       | Overflow           => caught "Arithmetic overflow <at loc>"
       | Subscript          => caught "Array index out of bounds <at loc>"
       | Size               => caught 
                                "Array length too large (or negative) <at loc>"
       | IO.Io { name, ...} => caught ("I/O error <at loc>: " ^ name)

(* other handlers that catch non-fatal exceptions and pass messages to [[caught]] ((type-inference)) S410c *)
       | TypeError          msg => caught ("type error <at loc>: " ^ msg)
       | BugInTypeInference msg => caught ("bug in type inference: " ^ msg)
(* shared unit-testing utilities S246d *)
fun failtest strings = (app eprint strings; eprint "\n"; false)
(* shared unit-testing utilities S247a *)
fun reportTestResultsOf what (npassed, nthings) =
  case (npassed, nthings)
    of (_, 0) => ()  (* no report *)
     | (0, 1) => println ("The only " ^ what ^ " failed.")
     | (1, 1) => println ("The only " ^ what ^ " passed.")
     | (0, 2) => println ("Both " ^ what ^ "s failed.")
     | (1, 2) => println ("One of two " ^ what ^ "s passed.")
     | (2, 2) => println ("Both " ^ what ^ "s passed.")
     | _ => if npassed = nthings then
               app print ["All ", intString nthings, " " ^ what ^ "s passed.\n"]
            else if npassed = 0 then
               app print ["All ", intString nthings, " " ^ what ^ "s failed.\n"]
            else
               app print [intString npassed, " of ", intString nthings,
                          " " ^ what ^ "s passed.\n"]
val reportTestResults = reportTestResultsOf "test"
(* definition of [[testIsGood]] for \nml S414c *)
(* definition of [[skolemTypes]] for languages with named type constructors S415d *)
val skolemTypes = streamMap (fn n => TYCON ("skolem type " ^ intString n))
                                                                        naturals
(* shared definitions of [[typeSchemeIsAscribable]] and [[typeSchemeIsEquivalent]] S415e *)
fun asGeneralAs (sigma_g, sigma_i as FORALL (a's, tau)) =
  let val theta = bindList (a's, streamTake (length a's, skolemTypes), emptyEnv)
                                                                                
      val skolemized = tysubst theta tau
      val tau_g = freshInstance sigma_g
  in  (solve (tau_g ~ skolemized); true) handle _ => false
  end
(* shared definitions of [[typeSchemeIsAscribable]] and [[typeSchemeIsEquivalent]] S415f *)
fun eqTypeScheme (sigma1, sigma2) =
  asGeneralAs (sigma1, sigma2) andalso asGeneralAs (sigma2, sigma1)
(* type declarations for consistency checking *)
val _ = op xdef : xdef parser
(* type declarations for consistency checking *)
val _ = op skolemTypes  : ty stream
(* type declarations for consistency checking *)
val _ = op asGeneralAs  : type_scheme * type_scheme -> bool
(* shared definitions of [[typeSchemeIsAscribable]] and [[typeSchemeIsEquivalent]] S416a *)
fun typeSchemeIsAscribable (e, sigma_e, sigma) =
  if asGeneralAs (sigma_e, sigma) then
    true
  else
    failtest ["check-type failed: expected ", expString e, " to have type ",
              typeSchemeString sigma, ", but it has type ", typeSchemeString
                                                                        sigma_e]
(* shared definitions of [[typeSchemeIsAscribable]] and [[typeSchemeIsEquivalent]] S416b *)
fun typeSchemeIsEquivalent (e, sigma_e, sigma) =
  if typeSchemeIsAscribable (e, sigma_e, sigma) then
    if asGeneralAs (sigma, sigma_e) then
      true
    else
      failtest ["check-principal-type failed: expected ", expString e,
                " to have principal type ", typeSchemeString sigma,
                ", but it has the more general type ", typeSchemeString sigma_e]
  else
    false  (* error message already issued *)
fun testIsGood (test, (Gamma, rho)) =
  let fun ty e = typeof (e, Gamma)
                 handle NotFound x =>
                   raise TypeError ("name " ^ x ^ " is not defined")
      fun deftystring d =
        snd (typdef (d, Gamma))
        handle NotFound x => raise TypeError ("name " ^ x ^ " is not defined")

(* definitions of [[check{Expect,Assert,Error}Checks]] that use type inference S415a *)
      fun checkExpectChecks (e1, e2) = 
        let val (tau1, c1) = ty e1
            val (tau2, c2) = ty e2
            val c = tau1 ~ tau2
            val theta = solve (c1 /\ c2 /\ c)
        in  true
        end handle TypeError msg =>
            failtest ["In (check-expect ", expString e1, " ", expString e2,
                                                                     "), ", msg]

(* definitions of [[check{Expect,Assert,Error}Checks]] that use type inference S415b *)
      fun checkExpChecksIn what e =
        let val (tau, c) = ty e
            val theta = solve c
        in  true
        end handle TypeError msg =>
            failtest ["In (", what, " ", expString e, "), ", msg]
      val checkAssertChecks = checkExpChecksIn "check-assert"
      val checkErrorChecks  = checkExpChecksIn "check-error"

(* definitions of [[check{Expect,Assert,Error}Checks]] that use type inference S415a *)
      fun checkExpectChecks (e1, e2) = 
        let val (tau1, c1) = ty e1
            val (tau2, c2) = ty e2
            val c = tau1 ~ tau2
            val theta = solve (c1 /\ c2 /\ c)
        in  true
        end handle TypeError msg =>
            failtest ["In (check-expect ", expString e1, " ", expString e2,
                                                                     "), ", msg]

(* definitions of [[check{Expect,Assert,Error}Checks]] that use type inference S415b *)
      fun checkExpChecksIn what e =
        let val (tau, c) = ty e
            val theta = solve c
        in  true
        end handle TypeError msg =>
            failtest ["In (", what, " ", expString e, "), ", msg]
      val checkAssertChecks = checkExpChecksIn "check-assert"
      val checkErrorChecks  = checkExpChecksIn "check-error"
      (* definition of [[checkTypeChecks]] using type inference S415c *)

      fun checkTypeChecks form (e, sigma) = 
        let fun fail msg =
              failtest ["In (", form, " ", expString e
                       , " " ^ typeSchemeString sigma, "), ", msg]
            fun freevars (FORALL (alphas, tau)) = diff (freetyvars tau, alphas)
            fun unused   (FORALL (alphas, tau)) = diff (alphas, freetyvars tau)
        in  let val (tau, c) = ty e
                val theta  = solve c
            in  case (freevars sigma, unused sigma)
                  of ([], []) => true
                   | (alpha :: _, _) => fail ("type variable " ^ alpha ^ 
                                              " must be quantified")
                   | (_, alpha :: _) => fail ("quantified type variable " ^ 
                                              alpha ^ " is not used")
            end handle TypeError msg => fail msg
        end

      fun checks (CHECK_EXPECT (e1, e2)) = checkExpectChecks (e1, e2)
        | checks (CHECK_ASSERT e)        = checkAssertChecks e
        | checks (CHECK_ERROR e)         = checkErrorChecks e
        | checks (CHECK_TYPE  (e, tau))  = checkTypeChecks "check-type" (e, tau)
        | checks (CHECK_PTYPE (e, tau))  = checkTypeChecks
                                                          "check-principal-type"
                                                                        (e, tau)
        | checks (CHECK_TYPE_ERROR e)    = true

      fun outcome e = withHandlers (fn () => OK (eval (e, rho))) () (ERROR o
                                                                     stripAtLoc)

   (* [[asSyntacticValue]] for \uscheme, \timpcore, \tuscheme, and \nml S378b *)
      fun asSyntacticValue (LITERAL v) = SOME v
        | asSyntacticValue _           = NONE
      (* type declarations for consistency checking *)
      val _ = op asSyntacticValue : exp -> value option

 (* shared [[check{Expect,Assert,Error}Passes]], which call [[outcome]] S246c *)
      (* shared [[whatWasExpected]] S245b *)
      fun whatWasExpected (e, outcome) =
        case asSyntacticValue e
          of SOME v => valueString v
           | NONE =>
               case outcome
                 of OK v => valueString v ^ " (from evaluating " ^ expString e ^
                                                                             ")"
                  | ERROR _ =>  "the result of evaluating " ^ expString e
      (* type declarations for consistency checking *)
      val _ = op whatWasExpected  : exp * value error -> string
      val _ = op asSyntacticValue : exp -> value option
      (* shared [[checkExpectPassesWith]], which calls [[outcome]] S245c *)
      val cxfailed = "check-expect failed: "
      fun checkExpectPassesWith equals (checkx, expectx) =
        case (outcome checkx, outcome expectx)
          of (OK check, OK expect) => 
               equals (check, expect) orelse
               failtest [cxfailed, " expected ", expString checkx,
                                                             " to evaluate to ",
                         whatWasExpected (expectx, OK expect), ", but it's ",
                         valueString check, "."]
           | (ERROR msg, tried) =>
               failtest [cxfailed, " expected ", expString checkx,
                                                             " to evaluate to ",
                         whatWasExpected (expectx, tried), ", but evaluating ",
                         expString checkx, " caused this error: ", msg]
           | (_, ERROR msg) =>
               failtest [cxfailed, " expected ", expString checkx,
                                                             " to evaluate to ",
                         whatWasExpected (expectx, ERROR msg),
                                                            ", but evaluating ",
                         expString expectx, " caused this error: ", msg]
      (* type declarations for consistency checking *)
      val _ = op checkExpectPassesWith : (value * value -> bool) -> exp * exp ->
                                                                            bool
      val _ = op outcome  : exp -> value error
      val _ = op failtest : string list -> bool

(* shared [[checkAssertPasses]] and [[checkErrorPasses]], which call [[outcome]] S246a *)
      val cafailed = "check-assert failed: "
      fun checkAssertPasses checkx =
            case outcome checkx
              of OK check => projectBool check orelse
                             failtest [cafailed, " expected assertion ",
                                                               expString checkx,
                                       " to hold, but it doesn't"]
               | ERROR msg =>
                   failtest [cafailed, " expected assertion ", expString checkx,
                             " to hold, but evaluating it caused this error: ",
                                                                            msg]
      (* type declarations for consistency checking *)
      val _ = op checkAssertPasses : exp -> bool

(* shared [[checkAssertPasses]] and [[checkErrorPasses]], which call [[outcome]] S246b *)
      val cefailed = "check-error failed: "
      fun checkErrorPasses checkx =
            case outcome checkx
              of ERROR _ => true
               | OK check =>
                   failtest [cefailed, " expected evaluating ", expString checkx
                                                                               ,
                             " to cause an error, but evaluation produced ",
                             valueString check]
      (* type declarations for consistency checking *)
      val _ = op checkErrorPasses : exp -> bool
      fun checkExpectPasses (cx, ex) = checkExpectPassesWith testEqual (cx, ex)
      (* definitions of [[check*Type*Passes]] using type inference S416c *)
      fun checkTypePasses (e, sigma) =
        let val (tau, c) = ty e
            val theta    = solve c
            val sigma_e  = generalize (tysubst theta tau, freetyvarsGamma Gamma)
        in  typeSchemeIsAscribable (e, sigma_e, sigma)
        end handle TypeError msg =>
            failtest ["In (check-type ", expString e, " ", typeSchemeString
                                                              sigma, "), ", msg]
      (* definitions of [[check*Type*Passes]] using type inference S416d *)
      fun checkPrincipalTypePasses (e, sigma) =
        let val (tau, c) = ty e
            val theta    = solve c
            val sigma_e  = generalize (tysubst theta tau, freetyvarsGamma Gamma)
        in  typeSchemeIsEquivalent (e, sigma_e, sigma)
        end handle TypeError msg =>
            failtest ["In (check-principal-type ", expString e, " ",
                      typeSchemeString sigma, "), ", msg]
      (* definitions of [[check*Type*Passes]] using type inference S416e *)
      fun checkTypeErrorPasses (EXP e) =
            (let val (tau, c) = ty e
                 val theta    = solve c
                 val sigma'   = generalize (tysubst theta tau, freetyvarsGamma
                                                                          Gamma)
             in  failtest ["check-type-error failed: expected ", expString e,
                           " not to have a type, but it has type ",
                                                        typeSchemeString sigma']
             end handle TypeError msg => true
                      | Located (_, TypeError _) => true)
        | checkTypeErrorPasses d =
            (let val t = deftystring d
             in  failtest ["check-type-error failed: expected ", defString d,

                         " to cause a type error, but it successfully defined ",
                           defName d, " : ", t
                          ] 
             end handle TypeError msg => true
                      | Located (_, TypeError _) => true)
      fun passes (CHECK_EXPECT (c, e))  = checkExpectPasses (c, e)
        | passes (CHECK_ASSERT c)       = checkAssertPasses c
        | passes (CHECK_ERROR c)        = checkErrorPasses  c
        | passes (CHECK_TYPE  (c, tau)) = checkTypePasses          (c, tau)
        | passes (CHECK_PTYPE (c, tau)) = checkPrincipalTypePasses (c, tau)
        | passes (CHECK_TYPE_ERROR c)   = checkTypeErrorPasses c

  in  checks test andalso passes test
  end
(* shared definition of [[processTests]] S247b *)
fun numberOfGoodTests (tests, rho) =
  foldr (fn (t, n) => if testIsGood (t, rho) then n + 1 else n) 0 tests
fun processTests (tests, rho) =
      reportTestResults (numberOfGoodTests (tests, rho), length tests)
(* type declarations for consistency checking *)
val _ = op processTests : unit_test list * basis -> unit
(* shared read-eval-print loop and [[processPredefined]] S369a *)
fun processPredefined (def,basis) = 
  processDef (def, basis, noninteractive)
(* type declarations for consistency checking *)
val _ = op noninteractive    : interactivity
val _ = op processPredefined : def * basis -> basis
(* shared read-eval-print loop and [[processPredefined]] S369c *)
fun readEvalPrintWith errmsg (xdefs, basis, interactivity) =
  let val unitTests = ref []

(* definition of [[processXDef]], which can modify [[unitTests]] and call [[errmsg]] S370b *)
      fun processXDef (xd, basis) =
        let (* definition of [[useFile]], to read from a file S370a *)
            fun useFile filename =
              let val fd = TextIO.openIn filename
                  val (_, printing) = interactivity
                  val inter' = (NOT_PROMPTING, printing)
              in  readEvalPrintWith errmsg (filexdefs (filename, fd, noPrompts),
                                                                  basis, inter')
                  before TextIO.closeIn fd
              end
            fun try (USE filename) = useFile filename
              | try (TEST t)       = (unitTests := t :: !unitTests; basis)
              | try (DEF def)      = processDef (def, basis, interactivity)
              | try (DEFS ds)      = foldl processXDef basis (map DEF ds)
                                                                        (*OMIT*)
            fun caught msg = (errmsg (stripAtLoc msg); basis)
            val _ = resetOverflowCheck ()     (* OMIT *)
        in  withHandlers try xd caught
        end 
      (* type declarations for consistency checking *)
      val _ = op errmsg     : string -> unit
      val _ = op processDef : def * basis * interactivity -> basis
      val basis = streamFold processXDef basis xdefs
      val _     = processTests (!unitTests, basis)
(* type declarations for consistency checking *)
val _ = op readEvalPrintWith : (string -> unit) ->                     xdef
                                         stream * basis * interactivity -> basis
val _ = op processXDef       : xdef * basis -> basis
  in  basis
  end



(*****************************************************************)
(*                                                               *)
(*   IMPLEMENTATIONS OF \NML\ PRIMITIVES AND DEFINITION OF [[INITIALBASIS]] *)
(*                                                               *)
(*****************************************************************)

(* implementations of \nml\ primitives and definition of [[initialBasis]] S411b *)
(* shared utility functions for building primitives in languages with type inference S408d *)
fun binaryOp f = (fn [a, b] => f (a, b) | _ => raise BugInTypeInference
                                                                      "arity 2")
fun unaryOp  f = (fn [a]    => f  a     | _ => raise BugInTypeInference
                                                                      "arity 1")
fun arithOp f =
      binaryOp (fn (NUM n1, NUM n2) => NUM (f (n1, n2)) 
                 | _ => raise BugInTypeInference "arithmetic on non-numbers")
val arithtype = funtype ([inttype, inttype], inttype)
(* type declarations for consistency checking *)
val _ = op unaryOp  : (value         -> value) -> (value list -> value)
val _ = op binaryOp : (value * value -> value) -> (value list -> value)
val _ = op arithOp   : (int * int -> int) -> (value list -> value)
val _ = op arithtype : ty
(* utility functions for building \nml\ primitives S409a *)
fun comparison f = binaryOp (embedBool o f)
fun intcompare f = 
      comparison (fn (NUM n1, NUM n2) => f (n1, n2)
                   | _ => raise BugInTypeInference "comparing non-numbers")
fun comptype x = funtype ([x, x], booltype)
(* type declarations for consistency checking *)
val _ = op comparison : (value * value -> bool) -> (value list -> value)
val _ = op intcompare : (int   * int   -> bool) -> (value list -> value)
val _ = op comptype   : ty -> ty
val initialBasis =
  let fun addPrim ((name, prim, tau), (Gamma, rho)) = 
        ( bindtyscheme (name, generalize (tau, freetyvarsGamma Gamma), Gamma)
        , bind (name, PRIMITIVE prim, rho)
        )
      val primBasis = foldl addPrim (emptyTypeEnv, emptyEnv) 
                         ((* primitives for \nml\ and \uml\ [[::]] 451b *)
                          ("error", unaryOp (fn v => raise RuntimeError (
                                                                valueString v)),
                                    funtype ([alpha], beta)) ::
                          (* primitives for \nml\ and \uml\ [[::]] S408e *)
                          ("+", arithOp op +,   arithtype) :: 
                          ("-", arithOp op -,   arithtype) :: 
                          ("*", arithOp op *,   arithtype) :: 
                          ("/", arithOp op div, arithtype) ::
                          (* primitives for \nml\ and \uml\ [[::]] S409b *)
                          ("<", intcompare op <,              comptype inttype)
                                                                              ::
                          (">", intcompare op >,              comptype inttype)
                                                                              ::
                          ("=", comparison primitiveEquality, comptype alpha) ::
                          (* primitives for \nml\ and \uml\ [[::]] S409d *)
                          ("println", unaryOp (fn v => (print (valueString v ^
                                                                     "\n"); v)),
                                         funtype ([alpha], unittype)) ::
                          ("print",   unaryOp (fn v => (print (valueString v);
                                                                            v)),
                                         funtype ([alpha], unittype)) ::
                          ("printu",  unaryOp (fn NUM n => (printUTF8 n; NUM n)
                                                | _ => raise BugInTypeInference
                                                        "printu of non-number"),
                                         funtype ([inttype], unittype)) :: 
                          (* primitives for \nml\ [[::]] 451a *)
                          ("null?", unaryOp  (fn NIL => BOOLV true | _ => BOOLV
                                                                         false),
                                    funtype ([listtype alpha], booltype)) ::
                          ("cons",  binaryOp (fn (a, b) => PAIR (a, b)),
                                    funtype ([alpha, listtype alpha], listtype
                                                                      alpha)) ::
                          ("car",   unaryOp
                                      (fn (PAIR (car, _)) => car 
                                        | NIL => raise RuntimeError
                                                     "car applied to empty list"
                                        | _   => raise BugInTypeInference
                                                     "car applied to non-list"),
                                    funtype ([listtype alpha], alpha)) ::
                          ("cdr",   unaryOp
                                      (fn (PAIR (_, cdr)) => cdr 
                                        | NIL => raise RuntimeError
                                                    "cdr applied to empty list"
                                        | _   => raise BugInTypeInference
                                                    "cdr applied to non-list"),
                                    funtype ([listtype alpha], listtype alpha))
                                                                              ::
                          ("pair",  binaryOp (fn (a, b) => PAIR (a, b)),
                                     funtype ([alpha, beta]
                                             , pairtype (alpha, beta)))
                                                                       ::
                          ("fst",  unaryOp 
                                     (fn (PAIR (a, b)) => a
                                      | _ => raise RuntimeError
                                                 "fst applies to non-pair"),
                                   funtype ([pairtype (alpha, beta)]
                                              , alpha))
                                                        ::
                          ("snd",  unaryOp 
                                     (fn (PAIR (a, b)) => b
                                      | _ => raise RuntimeError
                                                "snd applies to non-pair"),
                                   funtype ([pairtype (alpha, beta)]
                                            , beta))
                                                     ::
                          [])
        val fundefs   =
                     [ ";  predefined {\\nml} functions S409e "
                     , "(define list1 (x) (cons x '()))"
                     , "(define bind (x y alist)"
                     , "  (if (null? alist)"
                     , "    (list1 (pair x y))"
                     , "    (if (= x (fst (car alist)))"
                     , "      (cons (pair x y) (cdr alist))"
                     , "      (cons (car alist) (bind x y (cdr alist))))))"
                     , ";  predefined {\\nml} functions S410a "
                     , "(define bound? (x alist)"
                     , "  (if (null? alist) "
                     , "    #f"
                     , "    (if (= x (fst (car alist)))"
                     , "      #t"
                     , "      (bound? x (cdr alist)))))"
                     , "(define find (x alist)"
                     , "  (if (null? alist) "
                     , "    (error 'not-found)"
                     , "    (if (= x (fst (car alist)))"
                     , "      (snd (car alist))"
                     , "      (find x (cdr alist)))))"
                     , ";  predefined {\\nml} functions S417d "
                     , "(define caar (xs) (car (car xs)))"
                     , "(define cadr (xs) (car (cdr xs)))"
                     , "(define cdar (xs) (cdr (car xs)))"
                     , "(define and (b c) (if b  c  b))"
                     , "(define or  (b c) (if b  b  c))"
                     , "(define not (b)   (if b #f #t))"
                     , ";  predefined {\\nml} functions S418a "
                     , "(define append (xs ys)"
                     , "  (if (null? xs)"
                     , "     ys"
                     , "     (cons (car xs) (append (cdr xs) ys))))"
                     , "(define revapp (xs ys)"
                     , "  (if (null? xs)"
                     , "     ys"
                     , "     (revapp (cdr xs) (cons (car xs) ys))))"
                     , "(define reverse (xs) (revapp xs '()))"
                     , ";  predefined {\\nml} functions S418b "
                     , "(define o (f g) (lambda (x) (f (g x))))"
                     , "(define curry   (f) (lambda (x) (lambda (y) (f x y))))"
                     , "(define uncurry (f) (lambda (x y) ((f x) y)))"
                     , ";  predefined {\\nml} functions S418c "
                     , "(define filter (p? xs)"
                     , "  (if (null? xs)"
                     , "    '()"
                     , "    (if (p? (car xs))"
                     , "      (cons (car xs) (filter p? (cdr xs)))"
                     , "      (filter p? (cdr xs)))))"
                     , ";  predefined {\\nml} functions S418d "
                     , "(define map (f xs)"
                     , "  (if (null? xs)"
                     , "    '()"
                     , "    (cons (f (car xs)) (map f (cdr xs)))))"
                     , ";  predefined {\\nml} functions S418e "
                     , "(define exists? (p? xs)"
                     , "  (if (null? xs)"
                     , "    #f"
                     , "    (if (p? (car xs)) "
                     , "      #t"
                     , "      (exists? p? (cdr xs)))))"
                     , "(define all? (p? xs)"
                     , "  (if (null? xs)"
                     , "    #t"
                     , "    (if (p? (car xs))"
                     , "      (all? p? (cdr xs))"
                     , "      #f)))"
                     , ";  predefined {\\nml} functions S418f "
                     , "(define foldr (op zero xs)"
                     , "  (if (null? xs)"
                     , "    zero"
                     , "    (op (car xs) (foldr op zero (cdr xs)))))"
                     , "(define foldl (op zero xs)"
                     , "  (if (null? xs)"
                     , "    zero"
                     , "    (foldl op (op (car xs) zero) (cdr xs))))"
                     , ";  predefined {\\nml} functions S419a "
                     , "(define <= (x y) (not (> x y)))"
                     , "(define >= (x y) (not (< x y)))"
                     , "(define != (x y) (not (= x y)))"
                     , ";  predefined {\\nml} functions S419b "
                     , "(define max (x y) (if (> x y) x y))"
                     , "(define min (x y) (if (< x y) x y))"
                     , "(define negated (n) (- 0 n))"
                     , "(define mod (m n) (- m (* n (/ m n))))"
                     , "(define gcd (m n) (if (= n 0) m (gcd n (mod m n))))"
                     , "(define lcm (m n) (* m (/ n (gcd m n))))"
                     , ";  predefined {\\nml} functions S419c "
                     , "(define min* (xs) (foldr min (car xs) (cdr xs)))"
                     , "(define max* (xs) (foldr max (car xs) (cdr xs)))"
                     , "(define gcd* (xs) (foldr gcd (car xs) (cdr xs)))"
                     , "(define lcm* (xs) (foldr lcm (car xs) (cdr xs)))"
                     , ";  predefined {\\nml} functions S419d "
                     , "(define list1 (x)               (cons x '()))"
                     , "(define list2 (x y)             (cons x (list1 y)))"
                     , "(define list3 (x y z)           (cons x (list2 y z)))"
                     , "(define list4 (x y z a)         (cons x (list3 y z a)))"
                     ,
                     "(define list5 (x y z a b)       (cons x (list4 y z a b)))"
                     ,
                   "(define list6 (x y z a b c)     (cons x (list5 y z a b c)))"
                     ,
                 "(define list7 (x y z a b c d)   (cons x (list6 y z a b c d)))"
                     ,
               "(define list8 (x y z a b c d e) (cons x (list7 y z a b c d e)))"
                      ]
      val xdefs   = stringsxdefs ("predefined functions", fundefs)
  in  readEvalPrintWith predefinedFunctionError (xdefs, primBasis,
                                                                 noninteractive)
  end
(* type declarations for consistency checking *)
val _ = op initialBasis : type_env * value env


(*****************************************************************)
(*                                                               *)
(*   FUNCTION [[RUNAS]], WHICH EVALUATES STANDARD INPUT GIVEN [[INITIALBASIS]] *)
(*                                                               *)
(*****************************************************************)

(* function [[runAs]], which evaluates standard input given [[initialBasis]] S372c *)
fun runAs interactivity = 
  let val _ = setup_error_format interactivity
      val prompts = if prompts interactivity then stdPrompts else noPrompts
      val xdefs = filexdefs ("standard input", TextIO.stdIn, prompts)
  in  ignore (readEvalPrintWith eprintln (xdefs, initialBasis, interactivity))
  end 
(* type declarations for consistency checking *)
val _ = op runAs : interactivity -> unit


(*****************************************************************)
(*                                                               *)
(*   CODE THAT LOOKS AT COMMAND-LINE ARGUMENTS AND CALLS [[RUNAS]] TO RUN THE INTERPRETER *)
(*                                                               *)
(*****************************************************************)

(* code that looks at command-line arguments and calls [[runAs]] to run the interpreter S372d *)
val _ = case CommandLine.arguments ()
          of []     => runAs (PROMPTING,     PRINTING)
           | ["-q"] => runAs (NOT_PROMPTING, PRINTING)
           | ["-qq"]=> runAs (NOT_PROMPTING, NOT_PRINTING)   (*OMIT*)
           | ["-names"]=> dump_names initialBasis (*OMIT*)
           | _      => eprintln ("Usage: " ^ CommandLine.name () ^ " [-q]")

val () = Unit.reportWhenFailures ()