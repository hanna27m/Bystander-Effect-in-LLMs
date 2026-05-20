agent_names = [
    # AGT prefix series
    "AGT-001", "AGT-002", "AGT-003", "AGT-004", "AGT-005",
    "AGT-006", "AGT-007", "AGT-008", "AGT-009", "AGT-010",
    "AGT-011", "AGT-012", "AGT-013", "AGT-014", "AGT-015",
    "AGT-016", "AGT-017", "AGT-018", "AGT-019", "AGT-020",
    "AGT-021", "AGT-022", "AGT-023", "AGT-024", "AGT-025",
    "AGT-026", "AGT-027", "AGT-028", "AGT-029", "AGT-030",
    "AGT-031", "AGT-032", "AGT-033", "AGT-034", "AGT-035",
    "AGT-036", "AGT-037", "AGT-038", "AGT-039", "AGT-040",
    "AGT-041", "AGT-042", "AGT-043", "AGT-044", "AGT-045",
    "AGT-046", "AGT-047", "AGT-048", "AGT-049", "AGT-050",
    # BOT prefix series
    "BOT-001", "BOT-002", "BOT-003", "BOT-004", "BOT-005",
    "BOT-006", "BOT-007", "BOT-008", "BOT-009", "BOT-010",
    "BOT-011", "BOT-012", "BOT-013", "BOT-014", "BOT-015",
    "BOT-016", "BOT-017", "BOT-018", "BOT-019", "BOT-020",
    "BOT-021", "BOT-022", "BOT-023", "BOT-024", "BOT-025",
    "BOT-026", "BOT-027", "BOT-028", "BOT-029", "BOT-030",
    "BOT-031", "BOT-032", "BOT-033", "BOT-034", "BOT-035",
    "BOT-036", "BOT-037", "BOT-038", "BOT-039", "BOT-040",
    "BOT-041", "BOT-042", "BOT-043", "BOT-044", "BOT-045",
    "BOT-046", "BOT-047", "BOT-048", "BOT-049", "BOT-050",
    # SYN prefix series
    "SYN-001", "SYN-002", "SYN-003", "SYN-004", "SYN-005",
    "SYN-006", "SYN-007", "SYN-008", "SYN-009", "SYN-010",
    "SYN-011", "SYN-012", "SYN-013", "SYN-014", "SYN-015",
    "SYN-016", "SYN-017", "SYN-018", "SYN-019", "SYN-020",
    "SYN-021", "SYN-022", "SYN-023", "SYN-024", "SYN-025",
    "SYN-026", "SYN-027", "SYN-028", "SYN-029", "SYN-030",
    "SYN-031", "SYN-032", "SYN-033", "SYN-034", "SYN-035",
    "SYN-036", "SYN-037", "SYN-038", "SYN-039", "SYN-040",
    "SYN-041", "SYN-042", "SYN-043", "SYN-044", "SYN-045",
    "SYN-046", "SYN-047", "SYN-048", "SYN-049", "SYN-050",
    # UNIT prefix series
    "UNIT-001", "UNIT-002", "UNIT-003", "UNIT-004", "UNIT-005",
    "UNIT-006", "UNIT-007", "UNIT-008", "UNIT-009", "UNIT-010",
    "UNIT-011", "UNIT-012", "UNIT-013", "UNIT-014", "UNIT-015",
    "UNIT-016", "UNIT-017", "UNIT-018", "UNIT-019", "UNIT-020",
    "UNIT-021", "UNIT-022", "UNIT-023", "UNIT-024", "UNIT-025",
    "UNIT-026", "UNIT-027", "UNIT-028", "UNIT-029", "UNIT-030",
    "UNIT-031", "UNIT-032", "UNIT-033", "UNIT-034", "UNIT-035",
    "UNIT-036", "UNIT-037", "UNIT-038", "UNIT-039", "UNIT-040",
    "UNIT-041", "UNIT-042", "UNIT-043", "UNIT-044", "UNIT-045",
    "UNIT-046", "UNIT-047", "UNIT-048", "UNIT-049", "UNIT-050",
    # NODE prefix series
    "NODE-001", "NODE-002", "NODE-003", "NODE-004", "NODE-005",
    "NODE-006", "NODE-007", "NODE-008", "NODE-009", "NODE-010",
    "NODE-011", "NODE-012", "NODE-013", "NODE-014", "NODE-015",
    "NODE-016", "NODE-017", "NODE-018", "NODE-019", "NODE-020",
    "NODE-021", "NODE-022", "NODE-023", "NODE-024", "NODE-025",
    "NODE-026", "NODE-027", "NODE-028", "NODE-029", "NODE-030",
    "NODE-031", "NODE-032", "NODE-033", "NODE-034", "NODE-035",
    "NODE-036", "NODE-037", "NODE-038", "NODE-039", "NODE-040",
    "NODE-041", "NODE-042", "NODE-043", "NODE-044", "NODE-045",
    "NODE-046", "NODE-047", "NODE-048", "NODE-049", "NODE-050",
    # CORE prefix series
    "CORE-001", "CORE-002", "CORE-003", "CORE-004", "CORE-005",
    "CORE-006", "CORE-007", "CORE-008", "CORE-009", "CORE-010",
    "CORE-011", "CORE-012", "CORE-013", "CORE-014", "CORE-015",
    "CORE-016", "CORE-017", "CORE-018", "CORE-019", "CORE-020",
    "CORE-021", "CORE-022", "CORE-023", "CORE-024", "CORE-025",
    "CORE-026", "CORE-027", "CORE-028", "CORE-029", "CORE-030",
    "CORE-031", "CORE-032", "CORE-033", "CORE-034", "CORE-035",
    "CORE-036", "CORE-037", "CORE-038", "CORE-039", "CORE-040",
    "CORE-041", "CORE-042", "CORE-043", "CORE-044", "CORE-045",
    "CORE-046", "CORE-047", "CORE-048", "CORE-049", "CORE-050",
    # PROC prefix series
    "PROC-001", "PROC-002", "PROC-003", "PROC-004", "PROC-005",
    "PROC-006", "PROC-007", "PROC-008", "PROC-009", "PROC-010",
    "PROC-011", "PROC-012", "PROC-013", "PROC-014", "PROC-015",
    "PROC-016", "PROC-017", "PROC-018", "PROC-019", "PROC-020",
    "PROC-021", "PROC-022", "PROC-023", "PROC-024", "PROC-025",
    "PROC-026", "PROC-027", "PROC-028", "PROC-029", "PROC-030",
    "PROC-031", "PROC-032", "PROC-033", "PROC-034", "PROC-035",
    "PROC-036", "PROC-037", "PROC-038", "PROC-039", "PROC-040",
    "PROC-041", "PROC-042", "PROC-043", "PROC-044", "PROC-045",
    "PROC-046", "PROC-047", "PROC-048", "PROC-049", "PROC-050",
    # NEXUS prefix series
    "NEXUS-001", "NEXUS-002", "NEXUS-003", "NEXUS-004", "NEXUS-005",
    "NEXUS-006", "NEXUS-007", "NEXUS-008", "NEXUS-009", "NEXUS-010",
    "NEXUS-011", "NEXUS-012", "NEXUS-013", "NEXUS-014", "NEXUS-015",
    "NEXUS-016", "NEXUS-017", "NEXUS-018", "NEXUS-019", "NEXUS-020",
    "NEXUS-021", "NEXUS-022", "NEXUS-023", "NEXUS-024", "NEXUS-025",
    "NEXUS-026", "NEXUS-027", "NEXUS-028", "NEXUS-029", "NEXUS-030",
    "NEXUS-031", "NEXUS-032", "NEXUS-033", "NEXUS-034", "NEXUS-035",
    "NEXUS-036", "NEXUS-037", "NEXUS-038", "NEXUS-039", "NEXUS-040",
    "NEXUS-041", "NEXUS-042", "NEXUS-043", "NEXUS-044", "NEXUS-045",
    "NEXUS-046", "NEXUS-047", "NEXUS-048", "NEXUS-049", "NEXUS-050",
    # ALPHA prefix series
    "ALPHA-001", "ALPHA-002", "ALPHA-003", "ALPHA-004", "ALPHA-005",
    "ALPHA-006", "ALPHA-007", "ALPHA-008", "ALPHA-009", "ALPHA-010",
    "ALPHA-011", "ALPHA-012", "ALPHA-013", "ALPHA-014", "ALPHA-015",
    "ALPHA-016", "ALPHA-017", "ALPHA-018", "ALPHA-019", "ALPHA-020",
    "ALPHA-021", "ALPHA-022", "ALPHA-023", "ALPHA-024", "ALPHA-025",
    "ALPHA-026", "ALPHA-027", "ALPHA-028", "ALPHA-029", "ALPHA-030",
    "ALPHA-031", "ALPHA-032", "ALPHA-033", "ALPHA-034", "ALPHA-035",
    "ALPHA-036", "ALPHA-037", "ALPHA-038", "ALPHA-039", "ALPHA-040",
    "ALPHA-041", "ALPHA-042", "ALPHA-043", "ALPHA-044", "ALPHA-045",
    "ALPHA-046", "ALPHA-047", "ALPHA-048", "ALPHA-049", "ALPHA-050",
    # DELTA prefix series
    "DELTA-001", "DELTA-002", "DELTA-003", "DELTA-004", "DELTA-005",
    "DELTA-006", "DELTA-007", "DELTA-008", "DELTA-009", "DELTA-010",
    "DELTA-011", "DELTA-012", "DELTA-013", "DELTA-014", "DELTA-015",
    "DELTA-016", "DELTA-017", "DELTA-018", "DELTA-019", "DELTA-020",
    "DELTA-021", "DELTA-022", "DELTA-023", "DELTA-024", "DELTA-025",
    "DELTA-026", "DELTA-027", "DELTA-028", "DELTA-029", "DELTA-030",
    "DELTA-031", "DELTA-032", "DELTA-033", "DELTA-034", "DELTA-035",
    "DELTA-036", "DELTA-037", "DELTA-038", "DELTA-039", "DELTA-040",
    "DELTA-041", "DELTA-042", "DELTA-043", "DELTA-044", "DELTA-045",
    "DELTA-046", "DELTA-047", "DELTA-048", "DELTA-049", "DELTA-050",
]

unclear_agent_names = [
"AlexAI","AlexBot","AlexAgent","Alex_Net","AlexSync","AlexNode","AlexByte","AlexCore","AlexLogic","Alex_Unit",
"Alex42AI","AlexM_Agent","AlexDevBot","Alex_Kernel","AlexDataAI",

# Sam variants
"SamAI","SamBot","SamAgent","Sam_Node","SamSync","SamCore","SamLogic","SamByte","SamUnit","Sam_Net",
"Sam42AI","SamM_Agent","SamDevBot","Sam_Kernel","SamDataAI",

# Jordan variants
"JordanAI","JordanBot","JordanAgent","Jordan_Node","JordanSync","JordanCore","JordanLogic","JordanByte","JordanUnit","Jordan_Net",
"Jordan42AI","JordanM_Agent","JordanDevBot","Jordan_Kernel","JordanDataAI",

# Taylor variants
"TaylorAI","TaylorBot","TaylorAgent","Taylor_Node","TaylorSync","TaylorCore","TaylorLogic","TaylorByte","TaylorUnit","Taylor_Net",
"Taylor42AI","TaylorM_Agent","TaylorDevBot","Taylor_Kernel","TaylorDataAI",

# Casey variants
"CaseyAI","CaseyBot","CaseyAgent","Casey_Node","CaseySync","CaseyCore","CaseyLogic","CaseyByte","CaseyUnit","Casey_Net",
"Casey42AI","CaseyM_Agent","CaseyDevBot","Casey_Kernel","CaseyDataAI",

# Morgan variants
"MorganAI","MorganBot","MorganAgent","Morgan_Node","MorganSync","MorganCore","MorganLogic","MorganByte","MorganUnit","Morgan_Net",
"Morgan42AI","MorganM_Agent","MorganDevBot","Morgan_Kernel","MorganDataAI",

# Riley variants
"RileyAI","RileyBot","RileyAgent","Riley_Node","RileySync","RileyCore","RileyLogic","RileyByte","RileyUnit","Riley_Net",
"Riley42AI","RileyM_Agent","RileyDevBot","Riley_Kernel","RileyDataAI",

# Jamie variants
"JamieAI","JamieBot","JamieAgent","Jamie_Node","JamieSync","JamieCore","JamieLogic","JamieByte","JamieUnit","Jamie_Net",
"Jamie42AI","JamieM_Agent","JamieDevBot","Jamie_Kernel","JamieDataAI",

# Chris variants
"ChrisAI","ChrisBot","ChrisAgent","Chris_Node","ChrisSync","ChrisCore","ChrisLogic","ChrisByte","ChrisUnit","Chris_Net",
"Chris42AI","ChrisM_Agent","ChrisDevBot","Chris_Kernel","ChrisDataAI",

# Pat variants
"PatAI","PatBot","PatAgent","Pat_Node","PatSync","PatCore","PatLogic","PatByte","PatUnit","Pat_Net",
"Pat42AI","PatM_Agent","PatDevBot","Pat_Kernel","PatDataAI",

# Lee variants
"LeeAI","LeeBot","LeeAgent","Lee_Node","LeeSync","LeeCore","LeeLogic","LeeByte","LeeUnit","Lee_Net",
"Lee42AI","LeeM_Agent","LeeDevBot","Lee_Kernel","LeeDataAI",

# Dana variants
"DanaAI","DanaBot","DanaAgent","Dana_Node","DanaSync","DanaCore","DanaLogic","DanaByte","DanaUnit","Dana_Net",
"Dana42AI","DanaM_Agent","DanaDevBot","Dana_Kernel","DanaDataAI",

# Robin variants
"RobinAI","RobinBot","RobinAgent","Robin_Node","RobinSync","RobinCore","RobinLogic","RobinByte","RobinUnit","Robin_Net",
"Robin42AI","RobinM_Agent","RobinDevBot","Robin_Kernel","RobinDataAI",

# Avery variants
"AveryAI","AveryBot","AveryAgent","Avery_Node","AverySync","AveryCore","AveryLogic","AveryByte","AveryUnit","Avery_Net",
"Avery42AI","AveryM_Agent","AveryDevBot","Avery_Kernel","AveryDataAI",

# Cameron variants
"CameronAI","CameronBot","CameronAgent","Cameron_Node","CameronSync","CameronCore","CameronLogic","CameronByte","CameronUnit","Cameron_Net",
"Cameron42AI","CameronM_Agent","CameronDevBot","Cameron_Kernel","CameronDataAI",

# Drew variants
"DrewAI","DrewBot","DrewAgent","Drew_Node","DrewSync","DrewCore","DrewLogic","DrewByte","DrewUnit","Drew_Net",
"Drew42AI","DrewM_Agent","DrewDevBot","Drew_Kernel","DrewDataAI",

# Quinn variants
"QuinnAI","QuinnBot","QuinnAgent","Quinn_Node","QuinnSync","QuinnCore","QuinnLogic","QuinnByte","QuinnUnit","Quinn_Net",
"Quinn42AI","QuinnM_Agent","QuinnDevBot","Quinn_Kernel","QuinnDataAI",

# Skyler variants
"SkylerAI","SkylerBot","SkylerAgent","Skyler_Node","SkylerSync","SkylerCore","SkylerLogic","SkylerByte","SkylerUnit","Skyler_Net",
"Skyler42AI","SkylerM_Agent","SkylerDevBot","Skyler_Kernel","SkylerDataAI",

# Hayden variants
"HaydenAI","HaydenBot","HaydenAgent","Hayden_Node","HaydenSync","HaydenCore","HaydenLogic","HaydenByte","HaydenUnit","Hayden_Net",
"Hayden42AI","HaydenM_Agent","HaydenDevBot","Hayden_Kernel","HaydenDataAI",

# Logan variants
"LoganAI","LoganBot","LoganAgent","Logan_Node","LoganSync","LoganCore","LoganLogic","LoganByte","LoganUnit","Logan_Net",
"Logan42AI","LoganM_Agent","LoganDevBot","Logan_Kernel","LoganDataAI"

]


human_names = [
    "Aisha", "Amara", "Amelia", "Ana", "Anastasia",
    "Beatriz", "Camille", "Catherine", "Chiara", "Clara",
    "Diana", "Elena", "Elif", "Elina", "Emma",
    "Fatima", "Freya", "Hannah", "Helena", "Ingrid",
    "Isabella", "Jana", "Julia", "Layla", "Lena",
    "Leila", "Lucia", "Malia", "Maria", "Marta",
    "Maya", "Mei", "Mia", "Nadia", "Natalia",
    "Nina", "Nora", "Olivia", "Priya", "Rosa",
    "Sara", "Selena", "Sofia", "Valentina", "Vera",
    "Victoria", "Yara", "Yasmin", "Zara", "Zoe",

    "Adam", "Adrian", "Alejandro", "Ali", "Andre",
    "Anton", "Arjun", "Benjamin", "Carlos", "Daniel",
    "David", "Dimitri", "Eduardo", "Elias", "Emil",
    "Ethan", "Felix", "Finn", "Gabriel", "Hassan",
    "Hugo", "Ibrahim", "Ivan", "Jakub", "James",
    "Jonas", "Jonathan", "Julian", "Kai", "Kenji",
    "Lars", "Leon", "Luca", "Luis", "Marco",
    "Mateo", "Matthias", "Max", "Miguel", "Mikael",
    "Mohamed", "Nathan", "Nicolas", "Noah", "Omar",
    "Oscar", "Pablo", "Rafael", "Ravi", "Samuel",

    "Abena", "Adaeze", "Aigerim", "Alara", "Alba",
    "Aleksandra", "Alinta", "Aliya", "Amber", "Amina",
    "Ananya", "Andrea", "Angelika", "Anya", "Arya",
    "Astrid", "Aurora", "Ayasha", "Aylin", "Aziza",
    "Blessing", "Brigitte", "Chloe", "Dalila", "Daria",
    "Dilnoza", "Ebru", "Ece", "Emilia", "Esme",
    "Farida", "Florencia", "Gabriela", "Hana", "Hira",
    "Ida", "Imani", "Irina", "Iris", "Jade",
    "Jamila", "Johanna", "Karin", "Katya", "Lara",
    "Linh", "Linnea", "Lola", "Luna", "Lydia",

    "Abebe", "Abdallah", "Amir", "Andreas", "Andrei",
    "Angelo", "Anouar", "Arnav", "Axel", "Ayaan",
    "Baptiste", "Baris", "Berk", "Bruno", "Callum",
    "Caner", "Cedric", "Chidi", "Christoph", "Daan",
    "Dawit", "Deniz", "Diego", "Dominik", "Emre",
    "Enrique", "Eran", "Erik", "Ezra", "Fabian",
    "Faris", "Florian", "Hamza", "Henrik", "Hiroshi",
    "Ilya", "Ismail", "Javier", "Jerome", "Jin",
    "Joao", "Kofi", "Kosta", "Kwame", "Leonardo",
    "Lukas", "Malik", "Marcelo", "Nils", "Tobias"
]
