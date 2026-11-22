"""Fleet schema, classifications and category screening bounds."""
EXPENSE_FIELDS = (
    'labor_internal','labor_external','parts','tires','fuel_expense','lubricant','wash'
)
KMPL_BOUNDS = {
    'automovel': (1.,24.), 'picape':(1.,20.), 'van':(1.,20.),
    'caminhao':(.7,15.), 'onibus':(.7,12.), 'outros':(.5,30.),
    'desconhecido':(.5,30.)
}
GENERIC_EXACT = {
    'CAMINHAO','ONIBUS','VAN','MICRO ONIBUS','M POLO','MARCOPOLO',
    'COMIL','RODOVIARIO','ATHENAS','ALLEGRO','REBOQUE','TANQUE',
    'CAVALO MEC','BASCULANTE','TRUCK'
}
GENERIC_TOKENS = {'MARCOPOLO','COMIL','RODOVIARIO','ATHENAS','ALLEGRO'}
FLEET_COLUMNS = [
    'fleet_id','model','os_count','labor_internal','labor_external',
    'parts','tires','fuel_expense','lubricant','wash','km',
    'kmpl_reported','total_expense','cost_per_km'
]

